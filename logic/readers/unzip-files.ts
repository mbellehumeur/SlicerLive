/**
 * Minimal ZIP → File[] for local DICOM archives (store + deflate).
 * Reads the central directory so sizes are correct when local headers use data descriptors.
 * Uses DecompressionStream("deflate-raw") — no npm dependency.
 */

const enc = new TextDecoder("utf-8");

function u16(view: DataView, o: number): number {
  return view.getUint16(o, true);
}
function u32(view: DataView, o: number): number {
  return view.getUint32(o, true);
}

async function inflateRaw(bytes: Uint8Array): Promise<Uint8Array> {
  const ds = new DecompressionStream("deflate-raw");
  return new Uint8Array(
    await new Response(new Blob([bytes as BlobPart]).stream().pipeThrough(ds)).arrayBuffer(),
  );
}

/** Find end-of-central-directory and return central-directory offset + entry count. */
function findCentralDirectory(view: DataView): { offset: number; count: number } {
  const bytes = view.byteLength;
  // EOCD is at least 22 bytes; comment can push it back up to 64 KiB.
  const start = Math.max(0, bytes - 22 - 0xffff);
  for (let i = bytes - 22; i >= start; i--) {
    if (u32(view, i) !== 0x06054b50) continue;
    return { count: u16(view, i + 10), offset: u32(view, i + 16) };
  }
  throw new Error("ZIP end-of-central-directory not found");
}

/** Unpack a ZIP ArrayBuffer into File objects (directories skipped). */
export async function unzipToFiles(buf: ArrayBuffer): Promise<File[]> {
  const bytes = new Uint8Array(buf);
  const view = new DataView(buf);
  const { offset: cdStart, count } = findCentralDirectory(view);
  const files: File[] = [];
  let o = cdStart;
  for (let n = 0; n < count; n++) {
    if (o + 46 > bytes.length || u32(view, o) !== 0x02014b50) {
      throw new Error("ZIP central directory truncated or corrupt");
    }
    const method = u16(view, o + 10);
    const compSize = u32(view, o + 20);
    const uncompSize = u32(view, o + 24);
    const nameLen = u16(view, o + 28);
    const extraLen = u16(view, o + 30);
    const commentLen = u16(view, o + 32);
    const localOff = u32(view, o + 42);
    const name = enc.decode(bytes.subarray(o + 46, o + 46 + nameLen));
    o += 46 + nameLen + extraLen + commentLen;
    if (!name || name.endsWith("/")) continue;

    if (localOff + 30 > bytes.length || u32(view, localOff) !== 0x04034b50) {
      throw new Error(`ZIP local header missing for ${name}`);
    }
    const localNameLen = u16(view, localOff + 26);
    const localExtraLen = u16(view, localOff + 28);
    const dataStart = localOff + 30 + localNameLen + localExtraLen;
    const dataEnd = dataStart + compSize;
    if (dataEnd > bytes.length) throw new Error(`ZIP truncated (${name})`);
    const payload = bytes.subarray(dataStart, dataEnd);

    let raw: Uint8Array;
    if (method === 0) {
      raw = payload.slice();
    } else if (method === 8) {
      raw = await inflateRaw(payload);
    } else {
      throw new Error(`ZIP compression method ${method} not supported (${name})`);
    }
    if (uncompSize && raw.byteLength !== uncompSize) {
      throw new Error(
        `ZIP size mismatch for ${name}: got ${raw.byteLength}, expected ${uncompSize}`,
      );
    }
    const base = name.split("/").pop() || name;
    files.push(new File([raw as BlobPart], base));
  }
  if (!files.length) throw new Error("No files found in ZIP");
  return files;
}

/** Unpack a ZIP File into File objects. */
export async function unzipFileToFiles(file: File): Promise<File[]> {
  return unzipToFiles(await file.arrayBuffer());
}
