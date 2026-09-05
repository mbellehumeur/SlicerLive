// load dcmjs, retrying across CDN mirrors (jsdelivr/unpkg are occasionally flaky from some networks)
(function loadDcmjs() {
    // Host apps (e.g. SlicerRad) may preload a local/vendored dcmjs via a wrapper worker
    // (importScripts their copy first) — respect it instead of hitting the CDN.
    if (typeof dcmjs !== "undefined")
        return;
    const mirrors = [
        'https://cdn.jsdelivr.net/npm/dcmjs@0.41.0/build/dcmjs.min.js',
        'https://unpkg.com/dcmjs@0.41.0/build/dcmjs.min.js',
        'https://cdn.jsdelivr.net/npm/dcmjs@0.41.0/build/dcmjs.js',
        'https://unpkg.com/dcmjs@0.41.0/build/dcmjs.js',
    ];
    for (let i = 0; i < 12; i++) {
        try {
            importScripts(mirrors[i % mirrors.length]);
            return;
        }
        catch (e) { /* try next */ }
    }
    throw new Error('dcmjs: all CDN mirrors failed');
})();
const s3url = (b) => 'https://' + (b || 'idc-open-data') + '.s3.us-east-1.amazonaws.com/';
let CT_S3 = s3url(), SEG_S3 = s3url();
let MODNAME = 'image';
const post = (m, x) => self.postMessage(m, x || []);
const prog = (msg, frac) => post({ t: 'progress', msg, frac });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function fetchRetry(url, opts, tries = 6) {
    let err;
    for (let i = 0; i < tries; i++) {
        const ac = new AbortController(), to = setTimeout(() => ac.abort(), 20000);
        try {
            const r = await fetch(url, { ...(opts || {}), signal: ac.signal });
            if (!r.ok && r.status !== 206)
                throw new Error('HTTP ' + r.status);
            return r;
        }
        catch (e) {
            err = e;
            if (i < tries - 1)
                await sleep(Math.min(4000, 250 * 2 ** i) * (0.6 + 0.8 * Math.random()));
        }
        finally {
            clearTimeout(to);
        }
    }
    throw err;
}
function naturalize(buf) {
    const dd = dcmjs.data.DicomMessage.readFile(buf);
    return dcmjs.data.DicomMetaDictionary.naturalizeDataset(dd.dict);
}
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const lps2ras = (v) => [-v[0], -v[1], v[2]]; // DICOM LPS -> RAS
async function fetchBuf(key, base) { return (await fetchRetry((base || CT_S3) + key)).arrayBuffer(); }
function makeThumb(ds) {
    let pd = ds.PixelData;
    if (Array.isArray(pd))
        pd = pd[0];
    if (!pd)
        return null;
    const nx = ds.Columns, ny = ds.Rows, TW = 64, TH = Math.max(1, Math.round(64 * ny / nx));
    const px = ds.PixelRepresentation === 1 ? new Int16Array(pd) : new Uint16Array(pd);
    const slope = Number(ds.RescaleSlope ?? 1), inter = Number(ds.RescaleIntercept ?? 0);
    const lev = Number((Array.isArray(ds.WindowCenter) ? ds.WindowCenter[0] : ds.WindowCenter) ?? 40);
    const win = Number((Array.isArray(ds.WindowWidth) ? ds.WindowWidth[0] : ds.WindowWidth)) || 400;
    const lo = lev - win / 2, sc = 255 / win, rgba = new Uint8ClampedArray(TW * TH * 4);
    for (let ty = 0; ty < TH; ty++) {
        const sy = (ty * ny / TH) | 0;
        for (let tx = 0; tx < TW; tx++) {
            let g = (px[sy * nx + ((tx * nx / TW) | 0)] * slope + inter - lo) * sc;
            g = g < 0 ? 0 : g > 255 ? 255 : g;
            const o = (ty * TW + tx) * 4;
            rgba[o] = rgba[o + 1] = rgba[o + 2] = g;
            rgba[o + 3] = 255;
        }
    }
    return { w: TW, h: TH, rgba };
}
async function buildVolume(ctKeys) {
    const nz = ctKeys.length;
    post({ t: 'ctinfo', count: nz });
    // Practical progressive schedule:
    //  1) header-probe evenly spaced keys → provisional middle + ends (IPP·normal)
    //  2) mount + paint provisional middle; refine "best mid" as each new IPP arrives
    //  3) concurrent fetch prefers running mid, then center-out among known headers
    const decoded = []; // { sortKey, ipp, pixels, inst, key }
    let nx, ny, iop, rowDir, colDir, normal, ps, win, lev;
    let originIPP = null;
    let originSk = 0;
    let spacing = 1;
    let vol = null;
    let slicePix = 0;
    let ready = false;
    let done = 0;
    const isPET = MODNAME === 'PET';
    const CONC = 8;

    function extractPixels(ds) {
        const slope = Number(ds.RescaleSlope ?? 1), inter = Number(ds.RescaleIntercept ?? 0);
        let pd = ds.PixelData;
        if (Array.isArray(pd))
            pd = pd[0];
        const px = ds.PixelRepresentation === 1 ? new Int16Array(pd) : new Uint16Array(pd);
        const out = new Float32Array(ds.Rows * ds.Columns);
        for (let p = 0; p < out.length; p++)
            out[p] = px[p] * slope + inter;
        return out;
    }

    function provisionalIjkToRAS(sliceSpacing, origin) {
        const c0 = lps2ras(rowDir.map((v) => v * ps[1]));
        const c1 = lps2ras(colDir.map((v) => v * ps[0]));
        const c2 = lps2ras(normal.map((v) => v * sliceSpacing));
        const o = lps2ras(origin);
        return [c0[0], c1[0], c2[0], o[0], c0[1], c1[1], c2[1], o[1], c0[2], c1[2], c2[2], o[2], 0, 0, 0, 1];
    }

    /** Prefer a cheap header range; fall back to a full GET if the parse is incomplete. */
    async function fetchDicomMeta(key) {
        const HEAD = 256 << 10;
        try {
            const buf = await fetchRetry(CT_S3 + key, { headers: { Range: `bytes=0-${HEAD - 1}` } }).then((r) => r.arrayBuffer());
            const ds = naturalize(buf);
            if (!ds.ImagePositionPatient || !ds.ImageOrientationPatient || !ds.Rows || !ds.Columns)
                throw new Error('incomplete header');
            return ds;
        }
        catch (_) {
            return naturalize(await fetchBuf(key));
        }
    }

    function applyWlFrom(ds) {
        const mod = String(ds.Modality || MODNAME).toUpperCase();
        if (mod === 'CT') {
            win = Number((Array.isArray(ds.WindowWidth) ? ds.WindowWidth[0] : ds.WindowWidth) ?? 400);
            lev = Number((Array.isArray(ds.WindowCenter) ? ds.WindowCenter[0] : ds.WindowCenter) ?? 40);
        }
        else {
            win = 400;
            lev = 40;
        }
    }

    function publishReady() {
        if (ready)
            return;
        const empty = lev - win / 2 - 1;
        vol = new Float32Array(nx * ny * nz);
        vol.fill(empty);
        post({
            t: 'ctready',
            dims: [nx, ny, nz],
            ijkToRAS: provisionalIjkToRAS(spacing, originIPP),
            win,
            lev,
            dtype: 'float32',
        });
        ready = true;
    }

    function zFor(item) {
        const k = Math.round((item.sortKey - originSk) / Math.max(1e-6, spacing));
        return Math.max(0, Math.min(nz - 1, k));
    }

    function postSlice(z) {
        const slab = vol.slice(z * slicePix, (z + 1) * slicePix);
        post({ t: 'ctslab', zStart: z, zCount: 1, dims: [nx, ny, nz], vol: slab.buffer }, [slab.buffer]);
    }

    function rewriteAll() {
        vol.fill(lev - win / 2 - 1);
        for (const item of decoded) {
            const z = zFor(item);
            vol.set(item.pixels, z * slicePix);
        }
        const full = vol.slice();
        post({ t: 'ctslab', zStart: 0, zCount: nz, dims: [nx, ny, nz], vol: full.buffer }, [full.buffer]);
    }

    function maybeRefineOrigin(item) {
        if (item.sortKey >= originSk - 1e-3)
            return;
        originIPP = item.ipp;
        originSk = item.sortKey;
        post({ t: 'ctgeom', ijkToRAS: provisionalIjkToRAS(spacing, originIPP), win, lev });
        rewriteAll();
    }

    /** Pick SliceThickness vs SpacingBetweenSlices by closeness to an IPP-derived gap. */
    function spacingFromTags(ds, gapHint) {
        const st = Number(ds.SliceThickness);
        const sbs = Number(ds.SpacingBetweenSlices);
        const hasSt = Number.isFinite(st) && st > 0;
        const hasSbs = Number.isFinite(sbs) && sbs > 0;
        if (Number.isFinite(gapHint) && gapHint > 0 && hasSt && hasSbs) {
            return Math.abs(st - gapHint) <= Math.abs(sbs - gapHint) ? st : sbs;
        }
        if (hasSt)
            return st;
        if (hasSbs)
            return sbs;
        return Number.isFinite(gapHint) && gapHint > 0 ? gapHint : 1;
    }

    function medianGap(sortedKeys) {
        if (sortedKeys.length < 2)
            return NaN;
        const gaps = [];
        for (let i = 1; i < sortedKeys.length; i++) {
            const g = sortedKeys[i] - sortedKeys[i - 1];
            if (g > 1e-6)
                gaps.push(g);
        }
        if (!gaps.length)
            return NaN;
        gaps.sort((a, b) => a - b);
        return gaps[gaps.length >> 1];
    }

    /**
     * Sparse probes are not adjacent slices — median ΔIPP is often K× true spacing.
     * Divide out the integer stride when it matches a DICOM tag (or fall back to tag).
     */
    function perSliceSpacing(probeGap, tagSp) {
        if (Number.isFinite(tagSp) && tagSp > 0 && Number.isFinite(probeGap) && probeGap > 0) {
            const ratio = probeGap / tagSp;
            const k = Math.max(1, Math.round(ratio));
            if (Math.abs(ratio - k) <= 0.25 * k)
                return probeGap / k;
            // Prefer tag when the probe gap is not a clean multiple (irregular / bad sample).
            return tagSp;
        }
        if (Number.isFinite(probeGap) && probeGap > 0)
            return probeGap;
        if (Number.isFinite(tagSp) && tagSp > 0)
            return tagSp;
        return 1;
    }

    function ingest(ds, key) {
        const ipp = ds.ImagePositionPatient.map(Number);
        if (!ready) {
            // Fallback if probe failed: mount on first full decode.
            // dims = [Columns, Rows, N] — matches DICOM row-major + GPU bytesPerRow.
            nx = ds.Columns;
            ny = ds.Rows;
            slicePix = nx * ny;
            iop = ds.ImageOrientationPatient.map(Number);
            rowDir = iop.slice(0, 3);
            colDir = iop.slice(3, 6);
            normal = cross(rowDir, colDir);
            ps = ds.PixelSpacing.map(Number);
            spacing = spacingFromTags(ds, NaN);
            applyWlFrom(ds);
            originIPP = ipp;
            originSk = dot(ipp, normal);
            publishReady();
        }
        if (ds.Columns !== nx || ds.Rows !== ny)
            throw new Error('idc_tools: slice size ' + ds.Columns + 'x' + ds.Rows + ' != ' + nx + 'x' + ny);
        const sortKey = dot(ipp, normal);
        const inst = Number(ds.InstanceNumber);
        const pixels = extractPixels(ds);
        const item = { sortKey, ipp, pixels, inst, key };
        decoded.push(item);
        const z = zFor(item);
        vol.set(pixels, z * slicePix);
        postSlice(z);
        maybeRefineOrigin(item);
        const th = makeThumb(ds);
        if (th)
            post({ t: 'thumb', n: Number.isFinite(inst) ? inst : decoded.length, w: th.w, h: th.h, rgba: th.rgba.buffer }, [th.rgba.buffer]);
        done++;
        if (done % 4 === 0 || done === nz)
            prog(MODNAME + ' ' + done + '/' + nz, 0.05 + 0.5 * done / nz);
        return item;
    }

    // ---- Phase 1: header probe -----------------------------------------------------------
    prog('probing geometry…', 0.02);
    const nProbe = Math.min(16, nz);
    const probeIdx = [];
    const seenIdx = new Set();
    for (let i = 0; i < nProbe; i++) {
        const j = nProbe === 1 ? 0 : Math.round(i * (nz - 1) / (nProbe - 1));
        if (!seenIdx.has(j)) {
            seenIdx.add(j);
            probeIdx.push(j);
        }
    }
    const probes = [];
    await Promise.all(probeIdx.map(async (ji) => {
        const key = ctKeys[ji];
        const ds = await fetchDicomMeta(key);
        const ipp = ds.ImagePositionPatient.map(Number);
        const iop0 = ds.ImageOrientationPatient.map(Number);
        probes.push({ key, ds, ipp, iop: iop0, inst: Number(ds.InstanceNumber) });
    }));
    if (!probes.length)
        throw new Error('idc_tools: geometry probe returned no slices');

    // Shared orientation from the first probe; sortKey needs the normal.
    // dims = [Columns, Rows, N] (Cornerstone / demos/idc-worker.js convention).
    {
        const ds0 = probes[0].ds;
        iop = probes[0].iop;
        rowDir = iop.slice(0, 3);
        colDir = iop.slice(3, 6);
        normal = cross(rowDir, colDir);
        nx = ds0.Columns;
        ny = ds0.Rows;
        slicePix = nx * ny;
        ps = ds0.PixelSpacing.map(Number);
        applyWlFrom(ds0);
    }
    for (const p of probes)
        p.sortKey = dot(p.ipp, normal);
    probes.sort((a, b) => a.sortKey - b.sortKey);
    const lo = probes[0];
    const hi = probes[probes.length - 1];
    const mid = probes[probes.length >> 1];
    const probeGap = medianGap(probes.map((p) => p.sortKey));
    // Tag pick without the raw multi-slice probe gap (that inflated PET Z ~10×).
    const tagSp = spacingFromTags(mid.ds, NaN);
    spacing = perSliceSpacing(probeGap, tagSp);
    // Prefer the spatially lowest probed IPP as z=0 origin (refined as true min arrives).
    originIPP = lo.ipp;
    originSk = lo.sortKey;
    publishReady();
    prog('loading middle slice…', 0.04);

    // ---- Phase 2: full fetches; refine "middle" as new IPP headers arrive ---------------
    // knownSk grows with every header (probes + each full decode). bestMid = closest to
    // (min+max)/2 among known keys — bump that key to the front of the remaining queue.
    const knownSk = new Map(); // key -> sortKey
    for (const p of probes)
        knownSk.set(p.key, p.sortKey);

    function bestMidKey() {
        let loSk = Infinity, hiSk = -Infinity;
        for (const sk of knownSk.values()) {
            if (sk < loSk)
                loSk = sk;
            if (sk > hiSk)
                hiSk = sk;
        }
        if (!Number.isFinite(loSk) || !Number.isFinite(hiSk))
            return null;
        const target = (loSk + hiSk) / 2;
        let best = null, bestD = Infinity;
        for (const [k, sk] of knownSk) {
            const d = Math.abs(sk - target);
            if (d < bestD) {
                bestD = d;
                best = k;
            }
        }
        return best;
    }

    /** Prefer pending best-mid; else an unknown key (expands IPP range); else closest known. */
    function claimNext(pending) {
        if (!pending.size)
            return null;
        const prefer = bestMidKey();
        if (prefer && pending.has(prefer)) {
            pending.delete(prefer);
            return prefer;
        }
        // Expand known IPP span so the mid target improves before we drain the rest.
        for (const k of pending) {
            if (!knownSk.has(k)) {
                pending.delete(k);
                return k;
            }
        }
        let loSk = Infinity, hiSk = -Infinity;
        for (const sk of knownSk.values()) {
            if (sk < loSk)
                loSk = sk;
            if (sk > hiSk)
                hiSk = sk;
        }
        const target = Number.isFinite(loSk) ? (loSk + hiSk) / 2 : 0;
        let best = null, bestD = Infinity;
        for (const k of pending) {
            const d = Math.abs(knownSk.get(k) - target);
            if (d < bestD) {
                bestD = d;
                best = k;
            }
        }
        if (best) {
            pending.delete(best);
            return best;
        }
        return null;
    }

    const pending = new Set(ctKeys);

    // Paint provisional mid first (fast useful first paint from sparse probe).
    {
        pending.delete(mid.key);
        const midDs = naturalize(await fetchBuf(mid.key));
        const midIpp = midDs.ImagePositionPatient.map(Number);
        knownSk.set(mid.key, dot(midIpp, normal));
        ingest(midDs, mid.key);
    }
    // Far probe end: refine spacing, also expands known IPP range for mid estimate.
    if (hi.key !== mid.key && pending.has(hi.key)) {
        pending.delete(hi.key);
        const hiDs = naturalize(await fetchBuf(hi.key));
        const hiIpp = hiDs.ImagePositionPatient.map(Number);
        knownSk.set(hi.key, dot(hiIpp, normal));
        const hiItem = ingest(hiDs, hi.key);
        const midItem = decoded.find((d) => d.key === mid.key);
        if (midItem) {
            const dSk = Math.abs(hiItem.sortKey - midItem.sortKey);
            const dZ = Math.max(1, Math.round(dSk / Math.max(1e-6, spacing)));
            const refined = dSk / dZ;
            if (refined > 1e-4 && Math.abs(refined - spacing) / spacing > 0.05) {
                spacing = refined;
                post({ t: 'ctgeom', ijkToRAS: provisionalIjkToRAS(spacing, originIPP), win, lev });
                rewriteAll();
            }
        }
    }

    async function worker() {
        while (pending.size) {
            const key = claimNext(pending);
            if (!key)
                break;
            const ds = naturalize(await fetchBuf(key));
            const ipp = ds.ImagePositionPatient.map(Number);
            knownSk.set(key, dot(ipp, normal));
            ingest(ds, key);
        }
    }
    await Promise.all(Array.from({ length: Math.min(CONC, Math.max(1, nz)) }, worker));

    // Canonical z-order + geometry from full IPP sort.
    decoded.sort((a, b) => a.sortKey - b.sortKey);
    for (let z = 0; z < decoded.length; z++)
        vol.set(decoded[z].pixels, z * slicePix);
    const p0 = decoded[0].ipp;
    const p1 = decoded[nz - 1].ipp;
    const sliceSpacing = nz > 1 ? dot(sub(p1, p0), normal) / (nz - 1) : spacing;
    const ijkToRAS = provisionalIjkToRAS(sliceSpacing, p0);
    {
        const full = vol.slice();
        post({ t: 'ctslab', zStart: 0, zCount: nz, dims: [nx, ny, nz], vol: full.buffer }, [full.buffer]);
    }
    post({ t: 'ctgeom', ijkToRAS, win, lev });
    if (MODNAME !== 'CT') {
        const N = vol.length, step = Math.max(1, (N / 200000) | 0), samp = [];
        for (let i = 0; i < N; i += step) {
            const v = vol[i];
            if (!isPET || v > 0)
                samp.push(v);
        }
        samp.sort((a, b) => a - b);
        const pct = (f) => (samp.length ? samp[Math.min(samp.length - 1, (f * samp.length) | 0)] : 0);
        const loPct = isPET ? 0 : pct(0.01), hiPct = isPET ? (pct(0.98) || 1) : pct(0.99);
        lev = (loPct + hiPct) / 2;
        win = Math.max(1, hiPct - loPct);
    }
    return { vol, dims: [nx, ny, nz], ijkToRAS, win, lev, iop, ps, dtype: 'float32' };
}
function buildLabelmap(ds, bits, ct) {
    const [nx, ny, nz] = ct.dims;
    // SEG frames may differ from the source volume (e.g. 512² SEG on 128² PET).
    const segCols = Number(ds.Columns) || nx;
    const segRows = Number(ds.Rows) || ny;
    const frameBytes = (segCols * segRows) >> 3;
    const lab = new Uint8Array(nx * ny * nz);
    const M = ct.ijkToRAS, inv = invAffine(M);
    const toIJK = (lps) => {
        const r = lps2ras(lps);
        return [
            inv[0] * r[0] + inv[1] * r[1] + inv[2] * r[2] + inv[3],
            inv[4] * r[0] + inv[5] * r[1] + inv[6] * r[2] + inv[7],
            inv[8] * r[0] + inv[9] * r[1] + inv[10] * r[2] + inv[11]
        ];
    };
    const shared = ds.SharedFunctionalGroupsSequence?.[0] || {};
    const sIop = (shared.PlaneOrientationSequence?.[0]?.ImageOrientationPatient || ct.iop).map(Number);
    const sPs = (shared.PixelMeasuresSequence?.[0]?.PixelSpacing || ct.ps).map(Number);
    const colW = sIop.slice(0, 3).map((v) => v * sPs[1]);
    const rowW = sIop.slice(3, 6).map((v) => v * sPs[0]);
    const colors = [], names = {}, terminology = {};
    // one coded entry {scheme, value, meaning} from a DICOM code sequence's first item (or null)
    const code = (seq) => {
        const c = seq && seq[0];
        if (!c) return null;
        return { scheme: c.CodingSchemeDesignator || '', value: c.CodeValue || '', meaning: c.CodeMeaning || '' };
    };
    for (const s of (ds.SegmentSequence || [])) {
        const rgb = s.RecommendedDisplayCIELabValue ? dcmjs.data.Colors.dicomlab2RGB(s.RecommendedDisplayCIELabValue) : [1, 1, 1];
        const num = Number(s.SegmentNumber);
        colors.push([num, rgb[0], rgb[1], rgb[2]]);
        names[num] = s.SegmentLabel || ('Segment ' + s.SegmentNumber);
        // coded anatomical terminology (SegmentedPropertyType is the primary label; category/modifier/region add context)
        const type = s.SegmentedPropertyTypeCodeSequence;
        terminology[num] = {
            category: code(s.SegmentedPropertyCategoryCodeSequence),
            type: code(type),
            typeModifier: code(type && type[0] && type[0].SegmentedPropertyTypeModifierCodeSequence),
            region: code(s.AnatomicRegionSequence),
        };
    }
    const seenSeg = new Set();
    const perFrame = ds.PerFrameFunctionalGroupsSequence || [];
    const ref = (perFrame[0]?.PlanePositionSequence?.[0]?.ImagePositionPatient || [0, 0, 0]).map(Number), o0 = toIJK(ref);
    const diCol = sub(toIJK([ref[0] + colW[0], ref[1] + colW[1], ref[2] + colW[2]]), o0);
    const diRow = sub(toIJK([ref[0] + rowW[0], ref[1] + rowW[1], ref[2] + rowW[2]]), o0);
    for (let f = 0; f < perFrame.length; f++) {
        const fg = perFrame[f];
        const segNum = fg.SegmentIdentificationSequence?.[0]?.ReferencedSegmentNumber;
        const ippLps = fg.PlanePositionSequence?.[0]?.ImagePositionPatient?.map(Number);
        if (!segNum || !ippLps)
            continue;
        if (!seenSeg.has(segNum)) {
            seenSeg.add(segNum);
            post({ t: 'seg', name: names[segNum] || ('Segment ' + segNum) });
        }
        const o = toIJK(ippLps), fb = f * frameBytes;
        for (let row = 0; row < segRows; row++) {
            const bi = o[0] + row * diRow[0], bj = o[1] + row * diRow[1], bk = o[2] + row * diRow[2], rb = row * segCols;
            for (let col = 0; col < segCols; col++) {
                const p = rb + col;
                if (!((bits[fb + (p >> 3)] >> (p & 7)) & 1))
                    continue;
                const i = Math.round(bi + col * diCol[0]), j = Math.round(bj + col * diCol[1]), k = Math.round(bk + col * diCol[2]);
                if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz)
                    lab[k * nx * ny + j * nx + i] = segNum;
            }
        }
        if (f % 200 === 0)
            prog(`SEG ${f}/${perFrame.length}`, 0.55 + 0.4 * f / perFrame.length);
    }
    return { lab, colors, names, terminology };
}
function invAffine(m) {
    const a = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]], t = [m[3], m[7], m[11]];
    const det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
    const id = 1 / det;
    const r = [
        (a[4] * a[8] - a[5] * a[7]) * id, (a[2] * a[7] - a[1] * a[8]) * id, (a[1] * a[5] - a[2] * a[4]) * id,
        (a[5] * a[6] - a[3] * a[8]) * id, (a[0] * a[8] - a[2] * a[6]) * id, (a[2] * a[3] - a[0] * a[5]) * id,
        (a[3] * a[7] - a[4] * a[6]) * id, (a[1] * a[6] - a[0] * a[7]) * id, (a[0] * a[4] - a[1] * a[3]) * id
    ];
    const tx = -(r[0] * t[0] + r[1] * t[1] + r[2] * t[2]);
    const ty = -(r[3] * t[0] + r[4] * t[1] + r[5] * t[2]);
    const tz = -(r[6] * t[0] + r[7] * t[1] + r[8] * t[2]);
    return [r[0], r[1], r[2], tx, r[3], r[4], r[5], ty, r[6], r[7], r[8], tz, 0, 0, 0, 1];
}
async function fetchSeg(key) {
    const HEAD = 4 << 20;
    const head = new Uint8Array(await fetchRetry(SEG_S3 + key, { headers: { Range: `bytes=0-${HEAD - 1}` } }).then((r) => r.arrayBuffer()));
    const dv = new DataView(head.buffer, head.byteOffset);
    let pt = -1;
    for (let i = 132; i + 12 <= head.length; i += 2) {
        if (head[i] === 0xE0 && head[i + 1] === 0x7F && head[i + 2] === 0x10 && head[i + 3] === 0x00) {
            const vr = String.fromCharCode(head[i + 4], head[i + 5]);
            if (vr === 'OB' || vr === 'OW' || vr === 'UN') {
                pt = i;
                break;
            }
        }
    }
    if (pt < 0)
        throw new Error('PixelData tag not in header range');
    const valOff = pt + 12, pdLen = dv.getUint32(pt + 8, true);
    if (!pdLen || pdLen === 0xFFFFFFFF)
        throw new Error('encapsulated/undefined PixelData length');
    const ds = naturalize(head.slice(0, pt).buffer);
    const bits = new Uint8Array(pdLen);
    const have = Math.max(0, Math.min(HEAD, valOff + pdLen) - valOff);
    if (have > 0)
        bits.set(head.subarray(valOff, valOff + have), 0);
    const rs = valOff + have, re = valOff + pdLen - 1;
    if (rs <= re) {
        const CH = 8, cs = Math.ceil((re - rs + 1) / CH);
        let got = have;
        await Promise.all(Array.from({ length: CH }, (_, c) => {
            const s = rs + c * cs, e = Math.min(re, s + cs - 1);
            if (s > e)
                return null;
            return fetchRetry(SEG_S3 + key, { headers: { Range: `bytes=${s}-${e}` } }).then((r) => r.arrayBuffer()).then((ab) => {
                bits.set(new Uint8Array(ab), s - valOff);
                got += ab.byteLength;
                prog(`SEG ${(got / 1e6) | 0}/${(pdLen / 1e6) | 0} MB`, 0.5 + 0.08 * got / pdLen);
            });
        }));
    }
    return { ds, bits };
}
self.onmessage = async (e) => {
    const { ctKeys, segKeys, ctBucket, segBucket, modality } = e.data;
    CT_S3 = s3url(ctBucket);
    SEG_S3 = s3url(segBucket);
    MODNAME = { CT: 'CT', MR: 'MR', PT: 'PET' }[modality] || 'image';
    try {
        prog('fetching ' + MODNAME + '…', 0.05);
        const ct = await buildVolume(ctKeys);
        post({ t: 'ct', vol: ct.vol, dims: ct.dims, ijkToRAS: ct.ijkToRAS, win: ct.win, lev: ct.lev, dtype: ct.dtype }, [ct.vol.buffer]);
        if (segKeys && segKeys.length) {
            prog('fetching SEG…', 0.5);
            let parsed;
            try {
                parsed = await fetchSeg(segKeys[0]);
            }
            catch (err) {
                const buf = await fetchBuf(segKeys[0], SEG_S3);
                const ds = naturalize(buf);
                let pd = ds.PixelData;
                if (Array.isArray(pd))
                    pd = pd[0];
                parsed = { ds, bits: new Uint8Array(pd) };
            }
            const seg = buildLabelmap(parsed.ds, parsed.bits, ct);
            post({ t: 'labelmap', lab: seg.lab, colors: seg.colors, names: seg.names, terminology: seg.terminology }, [seg.lab.buffer]);
        }
        post({ t: 'alldone' });
    }
    catch (err) {
        post({ t: 'error', error: String(err && err.stack || err) });
    }
};
