/**
 * Nearest-neighbor resample of a multilabel volume onto a CT IJK grid via RAS.
 * Used by Cast NIfTI SEG and LiveScene `vtkMRMLSegmentationNode` labelmaps.
 */

/** Invert a row-major 4×4 affine (same layout as CTVolume / parseNifti ijkToRAS). */
function invAffineRowMajor(m: ArrayLike<number>): number[] {
  const a = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
  const t = [m[3], m[7], m[11]];
  const det =
    a[0]! * (a[4]! * a[8]! - a[5]! * a[7]!) -
    a[1]! * (a[3]! * a[8]! - a[5]! * a[6]!) +
    a[2]! * (a[3]! * a[7]! - a[4]! * a[6]!);
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12) {
    throw new Error("ijkToRAS is singular — cannot resample onto CT");
  }
  const id = 1 / det;
  const r = [
    (a[4]! * a[8]! - a[5]! * a[7]!) * id,
    (a[2]! * a[7]! - a[1]! * a[8]!) * id,
    (a[1]! * a[5]! - a[2]! * a[4]!) * id,
    (a[5]! * a[6]! - a[3]! * a[8]!) * id,
    (a[0]! * a[8]! - a[2]! * a[6]!) * id,
    (a[2]! * a[3]! - a[0]! * a[5]!) * id,
    (a[3]! * a[7]! - a[4]! * a[6]!) * id,
    (a[1]! * a[6]! - a[0]! * a[7]!) * id,
    (a[0]! * a[4]! - a[1]! * a[3]!) * id,
  ];
  return [
    r[0]!,
    r[1]!,
    r[2]!,
    -(r[0]! * t[0]! + r[1]! * t[1]! + r[2]! * t[2]!),
    r[3]!,
    r[4]!,
    r[5]!,
    -(r[3]! * t[0]! + r[4]! * t[1]! + r[5]! * t[2]!),
    r[6]!,
    r[7]!,
    r[8]!,
    -(r[6]! * t[0]! + r[7]! * t[1]! + r[8]! * t[2]!),
    0,
    0,
    0,
    1,
  ];
}

function applyRowMajor(
  m: ArrayLike<number>,
  i: number,
  j: number,
  k: number,
): [number, number, number] {
  return [
    m[0]! * i + m[1]! * j + m[2]! * k + m[3]!,
    m[4]! * i + m[5]! * j + m[6]! * k + m[7]!,
    m[8]! * i + m[9]! * j + m[10]! * k + m[11]!,
  ];
}

/**
 * Nearest-neighbor resample of multilabel volume onto the CT IJK grid via RAS.
 * Fixes TotalSeg / ITK axis flips vs IDC DICOM volumes that share dims but not index order.
 * Also used for LiveScene labelmaps (often a tight crop of the CT).
 */
export function resampleLabelmapToCt(
  src: ArrayLike<number>,
  srcDims: [number, number, number],
  srcIjkToRas: ArrayLike<number>,
  ctDims: [number, number, number],
  ctIjkToRas: ArrayLike<number>,
): { lab: Uint8Array; present: Set<number>; clipped: number } {
  const [snx, sny, snz] = srcDims;
  const [nx, ny, nz] = ctDims;
  const rasToSrc = invAffineRowMajor(srcIjkToRas);
  const lab = new Uint8Array(nx * ny * nz);
  const present = new Set<number>();
  let clipped = 0;

  for (let k = 0; k < nz; k++) {
    const plane = k * nx * ny;
    for (let j = 0; j < ny; j++) {
      const row = plane + j * nx;
      for (let i = 0; i < nx; i++) {
        const [rx, ry, rz] = applyRowMajor(ctIjkToRas, i, j, k);
        const [si, sj, sk] = applyRowMajor(rasToSrc, rx, ry, rz);
        const ii = Math.round(si);
        const jj = Math.round(sj);
        const kk = Math.round(sk);
        if (ii < 0 || jj < 0 || kk < 0 || ii >= snx || jj >= sny || kk >= snz) {
          continue;
        }
        let v = Math.round(Number(src[kk * snx * sny + jj * snx + ii]));
        if (!Number.isFinite(v) || v <= 0) continue;
        if (v > 255) {
          clipped++;
          v = 255;
        }
        lab[row + i] = v;
        present.add(v);
      }
    }
  }
  return { lab, present, clipped };
}
