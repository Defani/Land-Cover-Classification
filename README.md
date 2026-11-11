# Land Cover Classification

A [RustGIS](https://github.com/opengeos/RustGIS) plugin for supervised land cover
classification, in the spirit of Google Earth Engine's classifiers:

- **Draw training samples on the map** — points, rectangles and polygons, one class at a time,
  from the **Land Cover** panel. Samples go into the `Training Points` and `Training Polygons`
  layers, coloured by class, with a feature template per class so polygons can also be drawn
  with **Edit › Create**.
- **Composite Bands** — stack band files into one multi-band GeoTIFF, as ArcGIS's Composite
  Bands does, but bands of other cell sizes (Sentinel-2's 10, 20 and 60 m bands) are resampled
  onto the grid you choose.
- **Classify** with Random Forest, Support Vector Machine, CART, k-nearest neighbours, naive
  Bayes, minimum distance or maximum likelihood, with your own training/test split, number of
  trees and other parameters. Classify reads the band files directly: no composite needed.
- **Accuracy assessment** — confusion matrix, overall accuracy, Cohen's kappa, producer's and
  user's accuracy, F1, omission and commission errors, out-of-bag accuracy and variable
  importance (Random Forest), and the area of each class. The **Accuracy Assessment** tool also
  checks any classified map against independent reference samples.

Everything runs inside RustGIS's WebAssembly sandbox: no Python, no GDAL, no network.

## Using it

1. Add your image to the map (to see where to draw).
2. Open **Land Cover › Land Cover** (the panel). Choose a class under **Draw samples of**
   (Water, Vegetation, Built-up and Bare land to start with; **New Class** adds your own, with
   a colour), then click **Point**, **Rectangle** or **Polygon Vertex** and click (or drag) on
   the map. For a polygon, click **Polygon Vertex** then the map for each corner, then
   **Close Polygon**.
3. Set the classifier in the panel: algorithm, training share (70 = 70% train / 30% test),
   split by pixel or by whole sample, number of trees, variables per split, minimum leaf
   population, maximum depth, SVM kernel/C/gamma, k, maximum pixels per sample, seed.
4. Classify:
   - **Desktop:** the panel's **Classify** classifies the visible raster layers (or the one
     chosen under **Image**) read from their files, and adds the map.
   - **Web app** (or any time): **Land Cover › Classify** in the Tools pane: pick the band
     files (up to 12; a file may hold several bands), the training layers and the settings, run
     it, then click **Add Results to Map**.
5. Read the accuracy in the panel and in the tables the run adds (`… – Accuracy`,
   `… – Confusion Matrix`, `… – Class Accuracy`, `… – Variable Importance`, `… – Class Areas`).
   The map is a palette GeoTIFF (class colours and names included) you can export as it is.

### Large images run in passes

A RustGIS plugin tool may spend about 30 billion WebAssembly instructions per run. Training is
cheap (only the tiles under the samples are read), but running 100 trees over millions of
pixels is not, so a large map is classified in **passes**: the first run trains the model,
reports its accuracy and classifies the rows it has budget for, and leaves a
`… (classifying, NN% done)` layer. **Add Results to Map** runs the next passes (three per click,
since the desktop app stops a command after two minutes) and adds the map when it is
whole. Measured on a 2239 × 1348 Sentinel-2 scene (10 bands, 10 m): Random Forest with 100
trees took 7 passes, about 2½ minutes in total; at 20 m, or with 30 trees, it fits in one or
two.

Rough cost per pixel, cheapest first: minimum distance, naive Bayes ≪ maximum likelihood ≪ CART
≪ Random Forest (grows with trees) ≪ k-NN ≪ SVM with an RBF kernel (grows with support
vectors). A run that would need more than 40 passes is refused with advice: fewer trees, a
coarser grid, or clipping the rasters.

## Inputs

- GeoTIFF or BigTIFF, uncompressed, LZW, Deflate or PackBits, with or without a predictor;
  strips or tiles; bands interleaved or apart; 8–64-bit integers or floats; no-data from GDAL's
  tag. Files up to 64 MB each (RustGIS's limit for a file parameter).
- Coordinate systems: WGS 84 geographic, Web Mercator, UTM (WGS 84, NAD83, ETRS89, DGN95),
  DGN95 / Indonesia TM-3, or a Transverse Mercator the file defines. All rasters of a run must
  share one. Others: reproject first (**Analyze › Project Raster**).
- Training samples: point and polygon layers whose `class` field (or the field you name) holds
  a whole number 1–255 or a class name; a `label` field names the classes and a `color` field
  (`#rrggbb`) colours them. Pixels with no-data in any band, or zero in every band (the fill
  outside a Sentinel-2 or Landsat scene), are left out.

## The algorithms

| Algorithm | Parameters | Notes |
|---|---|---|
| Random Forest | trees, variables per split (√bands by default), min leaf, max depth, seed | Bootstrap samples, Gini splits on up to 256 quantile bins per band (Breiman 2001); out-of-bag accuracy and mean-decrease-in-impurity importance |
| CART | min leaf, max depth | One tree on every sample, all bands at each split (Breiman et al. 1984) |
| SVM | kernel (RBF, linear), C, gamma (1/bands by default) | C-SVC, one-against-one, SMO with second-order working-set selection (Fan, Chen & Lin 2005); bands standardised; each pair of classes trains on at most 1000 samples |
| k-NN | k | Bands standardised; k-d tree over at most 3000 samples |
| Naive Bayes | — | Gaussian, class priors from the samples |
| Minimum distance | — | Euclidean, to the class means |
| Maximum likelihood | — | Full covariance per class, equal priors; regularised when a class has fewer samples than bands |

Accuracy follows Congalton & Green: rows of the confusion matrix are the reference classes,
columns the classified ones; kappa is (pₒ − pₑ)/(1 − pₑ).

## Requirements

RustGIS newer than 0.6.0 for the panel above: it uses dropdowns, dialogs and picks on the map,
which RustGIS 0.6.0 doesn't know (it refuses the plugin). The [web app](https://web.rustgis.app)
has them today; the desktop app from its next release.

**For RustGIS 0.6.0** there is a variant, `land-cover-classification-<version>-rustgis-0.6.0.rgplugin`
(`python scripts/manifest.py --compat <folder>/plugin.json` writes its manifest; the module is
the same). Its panel has the same tools and settings in trees, sliders and checkboxes, and
samples are drawn with RustGIS's own tools:

1. Choose a class in the **Draw samples of** tree (13 common classes: Water, Vegetation,
   Built-up, Bare land, Mangrove, Cropland, Rice field, Plantation, Shrubland, Grassland,
   Wetland, Cloud, Shadow).
2. **Draw Points** or **Draw Polygons** turns on RustGIS's Create tool on the sample layer:
   click the map for points; click each corner and double-click for a polygon.
   **View as Rectangle** adds the map's view (zoomed in, under 5 km) as a rectangle.
3. Choose the next class: what you drew gets the class before it (so does Classify).

## Develop

You need [Rust](https://rustup.rs) with the `wasm32-unknown-unknown` target and Python 3.11+.

```sh
python scripts/manifest.py      # writes plugin.json (the tools' raster parameters are generated)
cargo test --release            # unit tests (classifiers, GeoTIFF reader/writer, projections, metrics)
python scripts/package.py       # builds and packages into dist/
```

The source: `src/tif.rs` (GeoTIFF reading by tile and writing), `src/proj.rs` (projections),
`src/stack.rs` (composites and resampling), `src/sample.rs` (sample pixels and the split),
`src/ml.rs` (the classifiers), `src/metrics.rs` (accuracy), `src/classify.rs` (the Classify tool
and its passes), `src/lib.rs` (the panel, the other tools and Add Results to Map).

## References

- Breiman, L. (2001). Random forests. *Machine Learning* 45, 5–32.
- Breiman, L., Friedman, J., Olshen, R., Stone, C. (1984). *Classification and Regression Trees*.
- Fan, R.-E., Chen, P.-H., Lin, C.-J. (2005). Working set selection using second order
  information for training support vector machines. *JMLR* 6, 1889–1918.
- Congalton, R. G., Green, K. (2019). *Assessing the Accuracy of Remotely Sensed Data*, 3rd ed.
- Karney, C. F. F. (2011). Transverse Mercator with an accuracy of a few nanometers.
  *Journal of Geodesy* 85, 475–485.

## License

MIT. The classifiers, GeoTIFF reader and projections are this plugin's own code, written from
the publications above. Its dependencies are `rustgis-plugin`, `serde_json` and `miniz_oxide`,
and theirs (`indexmap`, `hashbrown`, `foldhash`, `memchr`, `itoa`, `adler2`…), all under MIT,
Apache-2.0, Zlib or similar permissive licences.
