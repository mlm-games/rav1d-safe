//! Pre-split frame buffer for safe parallel decode.
//!
//! `PlaneRows` splits a contiguous frame buffer into per-row mutable slices,
//! enabling `split_at_mut`-based ownership splitting for tile and SB-row parallelism.
//!
//! Key design: "re-split, don't persist" — row slices are temporary views per phase,
//! re-created from the flat buffer. Tile recon gets column strips; filtering gets
//! full-width rows.

#![forbid(unsafe_code)]

/// Split a contiguous pixel buffer into per-row mutable slices.
///
/// Each row slice is exactly `width` pixels — the stride padding is excluded.
/// The returned `Vec` has exactly `height` entries, one per row.
///
/// # Panics
///
/// Panics if `stride < width` or if the buffer is too small for `height` rows.
pub fn split_into_rows<P>(
    buf: &mut [P],
    stride: usize,
    width: usize,
    height: usize,
) -> Vec<&mut [P]> {
    assert!(stride >= width, "stride ({stride}) < width ({width})");
    assert!(
        buf.len() >= stride * height.saturating_sub(1) + width,
        "buffer too small: len={} but need at least {} for {height} rows (stride={stride}, width={width})",
        buf.len(),
        stride * height.saturating_sub(1) + width,
    );
    buf.chunks_mut(stride)
        .take(height)
        .map(|row| &mut row[..width])
        .collect()
}

/// Split rows by tile column boundaries.
///
/// Consumes a `Vec` of row slices and distributes column sub-slices to per-tile Vecs.
/// Each tile gets its column range from `boundaries[t]..boundaries[t+1]`.
///
/// `boundaries` must be sorted, start at 0, and end at `row_width`.
/// E.g., for a 1920-wide frame with 4 tiles: `[0, 480, 960, 1440, 1920]`.
///
/// # Returns
///
/// `Vec<Vec<&mut [P]>>` — one inner `Vec` per tile, each containing one slice per row
/// covering that tile's column range.
///
/// # Panics
///
/// Panics if boundaries are not sorted, don't start at 0, or don't match row widths.
pub fn split_rows_by_tiles<'a, P>(
    rows: Vec<&'a mut [P]>,
    boundaries: &[usize],
) -> Vec<Vec<&'a mut [P]>> {
    assert!(boundaries.len() >= 2, "need at least 2 boundaries (start + end)");
    assert_eq!(boundaries[0], 0, "boundaries must start at 0");
    for w in boundaries.windows(2) {
        assert!(w[0] <= w[1], "boundaries must be sorted: {} > {}", w[0], w[1]);
    }

    let n_tiles = boundaries.len() - 1;
    let n_rows = rows.len();
    let mut tiles: Vec<Vec<&'a mut [P]>> = (0..n_tiles)
        .map(|_| Vec::with_capacity(n_rows))
        .collect();

    for mut row in rows {
        let row_len = row.len();
        let expected_width = *boundaries.last().unwrap();
        assert_eq!(
            row_len, expected_width,
            "row width ({row_len}) doesn't match boundaries end ({expected_width})"
        );

        for t in 0..n_tiles {
            let len = boundaries[t + 1] - boundaries[t];
            let (piece, rest) = row.split_at_mut(len);
            tiles[t].push(piece);
            row = rest;
        }
    }

    tiles
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_into_rows_basic() {
        let mut buf: Vec<u8> = (0..32).collect();
        let rows = split_into_rows(&mut buf, 8, 6, 4);
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0], &[0, 1, 2, 3, 4, 5]);
        assert_eq!(rows[1], &[8, 9, 10, 11, 12, 13]);
        assert_eq!(rows[2], &[16, 17, 18, 19, 20, 21]);
        assert_eq!(rows[3], &[24, 25, 26, 27, 28, 29]);
    }

    #[test]
    fn test_split_into_rows_width_equals_stride() {
        let mut buf: Vec<u8> = (0..16).collect();
        let rows = split_into_rows(&mut buf, 4, 4, 4);
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0], &[0, 1, 2, 3]);
        assert_eq!(rows[3], &[12, 13, 14, 15]);
    }

    #[test]
    fn test_split_rows_by_tiles_two_tiles() {
        let mut buf: Vec<u8> = (0..24).collect();
        let rows = split_into_rows(&mut buf, 6, 6, 4);
        let tiles = split_rows_by_tiles(rows, &[0, 3, 6]);

        assert_eq!(tiles.len(), 2);
        // Tile 0: columns 0..3
        assert_eq!(tiles[0].len(), 4);
        assert_eq!(tiles[0][0], &[0, 1, 2]);
        assert_eq!(tiles[0][1], &[6, 7, 8]);
        // Tile 1: columns 3..6
        assert_eq!(tiles[1].len(), 4);
        assert_eq!(tiles[1][0], &[3, 4, 5]);
        assert_eq!(tiles[1][1], &[9, 10, 11]);
    }

    #[test]
    fn test_split_rows_by_tiles_four_tiles() {
        let mut buf: Vec<u16> = (0..48).collect();
        let rows = split_into_rows(&mut buf, 12, 12, 4);
        let tiles = split_rows_by_tiles(rows, &[0, 3, 6, 9, 12]);

        assert_eq!(tiles.len(), 4);
        for tile in &tiles {
            assert_eq!(tile.len(), 4); // 4 rows each
            for row in tile {
                assert_eq!(row.len(), 3); // 3 columns each
            }
        }
        // Check first row values
        assert_eq!(tiles[0][0], &[0, 1, 2]);
        assert_eq!(tiles[1][0], &[3, 4, 5]);
        assert_eq!(tiles[2][0], &[6, 7, 8]);
        assert_eq!(tiles[3][0], &[9, 10, 11]);
    }

    #[test]
    fn test_split_rows_by_tiles_single_tile() {
        let mut buf: Vec<u8> = (0..16).collect();
        let rows = split_into_rows(&mut buf, 4, 4, 4);
        let tiles = split_rows_by_tiles(rows, &[0, 4]);

        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0].len(), 4);
        assert_eq!(tiles[0][0], &[0, 1, 2, 3]);
    }

    #[test]
    fn test_tile_writes_are_visible_in_original_buffer() {
        let mut buf: Vec<u8> = vec![0; 24];
        {
            let rows = split_into_rows(&mut buf, 6, 6, 4);
            let mut tiles = split_rows_by_tiles(rows, &[0, 3, 6]);

            // Write to tile 0, row 0, col 1
            tiles[0][0][1] = 42;
            // Write to tile 1, row 2, col 0
            tiles[1][2][0] = 99;
        }
        // Verify writes are visible in the original buffer
        assert_eq!(buf[1], 42);       // row 0, col 1
        assert_eq!(buf[2 * 6 + 3], 99); // row 2, col 3 (tile 1 col 0 = global col 3)
    }

    #[test]
    #[should_panic(expected = "stride")]
    fn test_split_into_rows_stride_too_small() {
        let mut buf: Vec<u8> = vec![0; 16];
        split_into_rows(&mut buf, 3, 4, 4);
    }

    #[test]
    #[should_panic(expected = "boundaries must start at 0")]
    fn test_split_rows_by_tiles_bad_boundaries() {
        let mut buf: Vec<u8> = vec![0; 16];
        let rows = split_into_rows(&mut buf, 4, 4, 4);
        split_rows_by_tiles(rows, &[1, 4]);
    }
}
