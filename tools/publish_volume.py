#!/usr/bin/env python3
"""Publish a Slicer volume to the SlicerLive gallery: Slicer -> zarr -> JS2 bucket.

    python3 tools/publish_volume.py CTACardio
    python3 tools/publish_volume.py CTAAbdomenPanoramix --node Panoramix-cropped

Why this exists
---------------
Porting a SlicerWGPU selftest means running it on THE SAME DATA the selftest loads
(CTACardio, MRHead, CTAAbdomenPanoramix, ...), not a synthetic stand-in. Those datasets
come from Slicer's SampleData, so the gallery needs them hosted somewhere CORS-enabled.

An older `live/publish.py` did this via a `mrml_sync` serializer that no longer exists on
this machine. This script is self-contained: it only produces what SlicerLive's loader
(`render/scene-volume.ts`) actually consumes —

  * chunked, DEFLATE-compressed zarr blobs  (zlib format == DecompressionStream('deflate'))
  * a multiscale pyramid under `<id>.zarr/<level>/k.j.i` (level 0 = full res) plus
    `attrs.zarrLevels` / `meta.json` so IRA can coarse→fine LOD
  * a {blobBase, nodes} wrapper carrying `zarr`, `dims`, `ijkToRAS`, window/level and the
    VR transfer functions (color / scalarOpacity / shade) read off the VolumePropertyNode

— which is exactly the structure of the existing MRHead.json.

Requirements
------------
* Slicer running with the slicer-mcp server (default http://localhost:2026/mcp).
* `openstack` configured with the project that OWNS the bucket. NOTE: the `slicerlive`
  container belongs to project **CIS230102_IU** (see live/publish.py). Other projects on
  this machine (BIO240357_IU, MED250016_IU) can READ it but get 403 on write.
* The container ACL already makes new objects public + CORS (`access-control-allow-origin: *`),
  so no per-object ACL work is needed.
"""
import argparse, json, os, subprocess, sys, urllib.request

MCP = os.environ.get("SLICER_MCP", "http://localhost:2026/mcp")
CLOUD = os.environ.get("OS_CLOUD", "CIS230102_IU")
CONTAINER = "slicerlive"
BUCKET_BASE = f"https://js2.jetstream-cloud.org:8001/swift/v1/{CONTAINER}"
LIVE = os.environ.get("SLICERLIVE_GALLERY", "/Users/pieper/slicer/live")


def mcp(code: str):
    """Call slicer-mcp execute_python and return the tool's text result."""
    req = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
           "params": {"name": "execute_python", "arguments": {"code": code}}}
    r = urllib.request.Request(MCP, data=json.dumps(req).encode(),
                               headers={"Content-Type": "application/json",
                                        "Accept": "application/json, text/event-stream"})
    with urllib.request.urlopen(r, timeout=1800) as f:
        body = f.read().decode()
    for line in body.splitlines():                     # tolerate SSE framing
        line = line[6:] if line.startswith("data: ") else line
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "result" in d:
            c = d["result"].get("content", [])
            return c[0].get("text", "") if c else ""
    return body


# Runs INSIDE Slicer. Kept as a string so the whole publish is one command.
EXPORT_SRC = r'''
import slicer, vtk, numpy as np, os, json, zlib, shutil

COARSE_MAX = 192  # stop pyramiding once the largest dim is this small (matches tools/zarr_convert.ts)

def _halve(arr):
    """2x box-downsample mean; arr is (K,J,I)."""
    nz, ny, nx = arr.shape
    oz, oy, ox = max(1, nz // 2), max(1, ny // 2), max(1, nx // 2)
    a = arr[:oz * 2, :oy * 2, :ox * 2].astype(np.float32, copy=False)
    return a.reshape(oz, 2, oy, 2, ox, 2).mean(axis=(1, 3, 5))

def _scaled_ijk(base, factor):
    m = list(base)
    for r in range(3):
        for c in range(3):
            m[r * 4 + c] *= factor
    return m

def _write_zarr_level(arr, zdir, chunks, dtype_str):
    """Write one level as deflate-padded chunks. arr (K,J,I). Returns (shape, chunkGrid, bytes)."""
    nz, ny, nx = arr.shape
    cz, cy, cx = chunks
    gz = (-(-nz // cz), -(-ny // cy), -(-nx // cx))
    os.makedirs(zdir, exist_ok=True)
    total = 0
    if dtype_str == "<i2":
        np_dtype = np.int16
    elif dtype_str == "<u1":
        np_dtype = np.uint8
    else:
        np_dtype = np.float32
    for kk in range(gz[0]):
        for jj in range(gz[1]):
            for ii in range(gz[2]):
                blk = np.zeros((cz, cy, cx), dtype=np_dtype)
                sub = arr[kk*cz:kk*cz+cz, jj*cy:jj*cy+cy, ii*cx:ii*cx+cx]
                # cast for coarser levels that may be float means of int16
                sub_c = np.ascontiguousarray(sub).astype(np_dtype, copy=False)
                blk[:sub_c.shape[0], :sub_c.shape[1], :sub_c.shape[2]] = sub_c
                raw = zlib.compress(blk.tobytes(), 6)
                open(os.path.join(zdir, "%d.%d.%d" % (kk, jj, ii)), "wb").write(raw)
                total += len(raw)
    return [nz, ny, nx], list(gz), total

def export_volume_scene(vol, outdir, node_id="vtkMRMLScalarVolumeNode1", chunks=(64,128,128)):
    if os.path.isdir(outdir): shutil.rmtree(outdir)
    arr0 = np.ascontiguousarray(slicer.util.arrayFromVolume(vol).astype(np.float32))  # (K,J,I)
    # Prefer int16 storage for CT-like ranges at full res; coarser levels stay float32 means → store as <i2 clipped
    full = np.ascontiguousarray(slicer.util.arrayFromVolume(vol).astype(np.int16))
    levels_arr = [full]
    cur = arr0
    while max(cur.shape) > COARSE_MAX:
        cur = _halve(cur)
        levels_arr.append(np.ascontiguousarray(np.round(cur).astype(np.int16)))

    m = vtk.vtkMatrix4x4(); vol.GetIJKToRASMatrix(m)
    ijk0 = [m.GetElement(r, c) for r in range(4) for c in range(4)]
    cz, cy, cx = chunks
    zarr_levels = []
    meta_levels = []
    total_all = 0
    for lev, arr in enumerate(levels_arr):
        zdir = os.path.join(outdir, node_id + ".zarr", str(lev))
        shape, gz, total = _write_zarr_level(arr, zdir, chunks, "<i2")
        total_all += total
        nz, ny, nx = shape
        desc = {"level": lev, "dir": node_id + ".zarr", "dataset": str(lev),
                "shape": shape, "chunks": [cz, cy, cx], "chunkGrid": gz,
                "dtype": "<i2", "bytes": total,
                "ijkToRAS": _scaled_ijk(ijk0, 2 ** lev), "dims": [nx, ny, nz]}
        zarr_levels.append(desc)
        meta_levels.append(desc)

    disp = vol.GetDisplayNode()
    nodes = {"vtkMRMLScalarVolumeDisplayNode1": {
        "id":"vtkMRMLScalarVolumeDisplayNode1","class":"vtkMRMLScalarVolumeDisplayNode",
        "name":"VolumeDisplay","refs":{},"blobs":{},
        "attrs":{"visibility":1,"window":float(disp.GetWindow()),"level":float(disp.GetLevel())}}}
    refs_display = ["vtkMRMLScalarVolumeDisplayNode1"]
    vrDisp = None
    for i in range(vol.GetNumberOfDisplayNodes()):
        d = vol.GetNthDisplayNode(i)
        if d and d.IsA("vtkMRMLVolumeRenderingDisplayNode"): vrDisp = d
    if vrDisp is not None and vrDisp.GetVolumePropertyNode():
        vp = vrDisp.GetVolumePropertyNode(); prop = vp.GetVolumeProperty()
        cf, of = prop.GetRGBTransferFunction(), prop.GetScalarOpacity()
        color = []
        for i in range(cf.GetSize()):
            v=[0]*6; cf.GetNodeValue(i,v); color.append([v[0],v[1],v[2],v[3]])
        sop = []
        for i in range(of.GetSize()):
            v=[0]*4; of.GetNodeValue(i,v); sop.append([v[0],v[1]])
        nodes["vtkMRMLVolumePropertyNode1"] = {"id":"vtkMRMLVolumePropertyNode1",
            "class":"vtkMRMLVolumePropertyNode","name":vp.GetName(),"refs":{},"blobs":{},
            "attrs":{"shade":int(prop.GetShade()),"color":color,"scalarOpacity":sop}}
        nodes["vtkMRMLGPURayCastVolumeRenderingDisplayNode1"] = {
            "id":"vtkMRMLGPURayCastVolumeRenderingDisplayNode1",
            "class":"vtkMRMLGPURayCastVolumeRenderingDisplayNode","name":"VolumeRendering",
            "refs":{"volumeProperty":["vtkMRMLVolumePropertyNode1"]},"blobs":{},
            "attrs":{"visibility":1,"kind":"volumeRendering"}}
        refs_display.append("vtkMRMLGPURayCastVolumeRenderingDisplayNode1")
    # Back-compat: attrs.zarr describes level 0; attrs.zarrLevels enables coarse→fine LOD in IRA.
    L0 = zarr_levels[0]
    nodes[node_id] = {"id":node_id,"class":"vtkMRMLScalarVolumeNode","name":vol.GetName(),
        "refs":{"display":refs_display},"blobs":{},
        "attrs":{"zarr":{"dir":L0["dir"],"dataset":L0["dataset"],"shape":L0["shape"],
                          "chunks":L0["chunks"],"chunkGrid":L0["chunkGrid"],"dtype":L0["dtype"],
                          "bytes":L0["bytes"]},
                 "zarrLevels": zarr_levels,
                 "dims":L0["dims"],"comps":1,"ijkToRAS":ijk0}}
    json.dump({"blobBase":"REPLACE_ME","nodes":nodes}, open(os.path.join(outdir,"scene.json"),"w"))
    meta = {"name": vol.GetName(), "levels": meta_levels, "ijkToRAS": ijk0, "dims": L0["dims"],
            "zarr": {"dir":L0["dir"],"dataset":L0["dataset"],"shape":L0["shape"],
                     "chunks":L0["chunks"],"chunkGrid":L0["chunkGrid"],"dtype":L0["dtype"]},
            "bytes": L0["bytes"]}
    json.dump(meta, open(os.path.join(outdir, "meta.json"), "w"))
    return {"levels": len(zarr_levels), "chunks": sum(L["chunkGrid"][0]*L["chunkGrid"][1]*L["chunkGrid"][2] for L in zarr_levels),
            "compressedMB": round(total_all/1e6,1), "dims": L0["dims"]}
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sample", help="SampleData name, e.g. CTACardio / MRHead / CTAAbdomenPanoramix")
    ap.add_argument("--node", help="MRML node name if it differs from the sample name")
    ap.add_argument("--name", help="bucket/scene name (default: the sample name)")
    ap.add_argument("--stage", default="/tmp/publish", help="local staging dir")
    args = ap.parse_args()

    name = args.name or args.sample
    outdir = os.path.join(args.stage, name)

    print(f"[1/4] downloading + exporting {args.sample} in Slicer …")
    code = EXPORT_SRC + f'''
import SampleData, json
vol = slicer.util.getFirstNodeByName({(args.node or args.sample)!r})
if vol is None:
    vol = SampleData.SampleDataLogic().downloadSample({args.sample!r})
vrLogic = slicer.modules.volumerendering.logic()
d = vrLogic.CreateDefaultVolumeRenderingNodes(vol); d.SetVisibility(True)
__result = json.dumps(export_volume_scene(vol, {outdir!r}))
'''
    print("   ", mcp(code))

    print(f"[2/4] uploading -> {CLOUD}:{CONTAINER}/{name}/blobs/ …")
    files = [os.path.join(r, f) for r, _, fs in os.walk(outdir) for f in fs if f != "scene.json"]
    for i, f in enumerate(files, 1):
        rel = os.path.relpath(f, outdir).replace(os.sep, "/")
        subprocess.run(["openstack", "--os-cloud", CLOUD, "object", "create", CONTAINER, rel,
                        "--name", f"{name}/blobs/{rel}"], cwd=outdir, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if i % 16 == 0 or i == len(files):
            print(f"    {i}/{len(files)}")

    print("[3/4] writing the scene wrapper …")
    w = json.load(open(os.path.join(outdir, "scene.json")))
    w["blobBase"] = f"{BUCKET_BASE}/{name}/blobs/"
    dst = os.path.join(LIVE, "scenes", f"{name}.json")
    json.dump(w, open(dst, "w"))
    print("    ", dst)

    print("[4/4] verifying the bucket is readable + CORS …")
    url = f"{BUCKET_BASE}/{name}/blobs/vtkMRMLScalarVolumeNode1.zarr/0/0.0.0"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            print(f"     HTTP {r.status}  CORS={r.headers.get('access-control-allow-origin')}")
    except Exception as e:
        sys.exit(f"     FAILED: {e}")
    print(f"\ndone — load it with ?scene={BUCKET_BASE.rsplit('/',1)[0]}/... or "
          f"https://pieper.github.io/live/scenes/{name}.json")


if __name__ == "__main__":
    main()
