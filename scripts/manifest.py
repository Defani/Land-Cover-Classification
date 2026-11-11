#!/usr/bin/env python3
"""Writes plugin.json: the tools' many raster parameters are generated here rather than typed."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = "0.1.0"


def raster_params(n, required=True):
    out = []
    for i in range(1, n + 1):
        p = {"name": f"raster{i}", "label": f"Raster {i}", "kind": "file", "extensions": ["tif", "tiff"], "required": required and i == 1}
        if i == 1:
            p["help"] = (
                "A GeoTIFF of one band (S2A_B4.tif) or several (a stack). Rasters of other cell sizes are "
                "resampled onto the grid below; all must be in one coordinate system."
            )
        out.append(p)
    return out


GRID = {
    "name": "grid",
    "label": "Grid",
    "kind": "choice",
    "choices": ["Finest raster", "Coarsest raster", "First raster"],
    "default": "Finest raster",
    "help": "Whose cells the result has: the finest (10 m for Sentinel-2's B2, B3, B4 and B8), the coarsest, or the first raster's.",
}

CLASSIFY = raster_params(12) + [
    GRID,
    {"name": "trainingPoints", "label": "Training points", "kind": "pointLayer",
     "help": "Point samples, each with its class (the Land Cover panel draws them into 'Training Points')."},
    {"name": "trainingPolygons", "label": "Training polygons", "kind": "polygonLayer",
     "help": "Rectangle and polygon samples (the panel draws them into 'Training Polygons')."},
    {"name": "classField", "label": "Class field", "kind": "text", "default": "class",
     "help": "The field holding each sample's class: a whole number 1-255, or a name. A 'label' field names the classes and a 'color' field (#rrggbb) colours them."},
    {"name": "algorithm", "label": "Algorithm", "kind": "choice",
     "choices": ["Random Forest", "Support Vector Machine", "CART (Decision Tree)", "k-Nearest Neighbors", "Naive Bayes", "Minimum Distance", "Maximum Likelihood"],
     "default": "Random Forest"},
    {"name": "splitRatio", "label": "Training share", "kind": "number", "default": 0.7,
     "help": "The share of each class's sample pixels used to train (0.7 = 70% training, 30% test). 1 trains on all of them and reports the training accuracy only."},
    {"name": "splitBy", "label": "Split by", "kind": "choice", "choices": ["Pixel", "Sample (whole polygon)"], "default": "Pixel",
     "help": "By pixel (as GEE's randomColumn) or by whole sample, so one polygon's pixels are never in both sets."},
    {"name": "trees", "label": "Number of trees", "kind": "number", "default": 100, "help": "Random Forest."},
    {"name": "variablesPerSplit", "label": "Variables per split", "kind": "number", "default": 0,
     "help": "Random Forest: bands tried at each split; 0 = the square root of the number of bands."},
    {"name": "minLeaf", "label": "Min leaf population", "kind": "number", "default": 1, "help": "Random Forest and CART."},
    {"name": "maxDepth", "label": "Max depth", "kind": "number", "default": 0, "help": "Random Forest and CART; 0 = unlimited."},
    {"name": "svmKernel", "label": "SVM kernel", "kind": "choice", "choices": ["RBF", "Linear"], "default": "RBF"},
    {"name": "svmCost", "label": "SVM cost (C)", "kind": "number", "default": 10},
    {"name": "svmGamma", "label": "SVM gamma", "kind": "number", "default": 0, "help": "0 = 1 / number of bands (bands are standardised first)."},
    {"name": "k", "label": "k (neighbours)", "kind": "number", "default": 5, "help": "k-Nearest Neighbors."},
    {"name": "maxPixels", "label": "Max pixels per sample", "kind": "number", "default": 500,
     "help": "A random choice of a large polygon's pixels, so big polygons don't swamp small ones."},
    {"name": "seed", "label": "Seed", "kind": "number", "default": 42, "help": "The same seed gives the same split and model."},
    {"name": "name", "label": "Output name", "kind": "text", "help": "Default: Land Cover (<algorithm>)."},
    {"name": "job", "label": "Continue a classification", "kind": "vectorLayer",
     "help": "Leave empty. A large map is classified in several runs: Add Results to Map continues it with the '(classifying)' layer the first run left."},
]

COMPOSITE = raster_params(24) + [
    GRID,
    {"name": "resampling", "label": "Resampling", "kind": "choice", "choices": ["Nearest", "Bilinear"], "default": "Nearest",
     "help": "For rasters whose cells differ from the grid's."},
    {"name": "name", "label": "Output name", "kind": "text", "default": "Composite"},
]

ASSESS = [
    {"name": "classified", "label": "Classified raster", "kind": "file", "extensions": ["tif", "tiff"], "required": True,
     "help": "A one-band class map (GeoTIFF)."},
    {"name": "reference", "label": "Reference samples", "kind": "vectorLayer", "required": True,
     "help": "Points or polygons whose class is known, drawn apart from the training samples."},
    {"name": "classField", "label": "Class field", "kind": "text", "default": "class", "help": "Its values must be the class map's values."},
    {"name": "maxPixels", "label": "Max pixels per sample", "kind": "number", "default": 500},
    {"name": "name", "label": "Output name", "kind": "text", "default": "Accuracy Assessment"},
]

ALGORITHMS = ["Random Forest", "Support Vector Machine", "CART (Decision Tree)", "k-Nearest Neighbors", "Naive Bayes", "Minimum Distance", "Maximum Likelihood"]

DESCRIPTION = (
    "Supervised land cover classification, as in Google Earth Engine: draw training samples on the map as points, "
    "rectangles or polygons per class, stack band files of any cell size (Composite Bands), classify with Random Forest, "
    "SVM, CART, k-NN, naive Bayes, minimum distance or maximum likelihood with your own training/test split and "
    "parameters, and assess the result with a confusion matrix, overall accuracy, kappa, producer's and user's accuracy, "
    "F1 and class areas."
)

MANIFEST = {
    "id": "land-cover-classification",
    "name": "Land Cover Classification",
    "version": VERSION,
    "description": DESCRIPTION,
    "author": "Defani Arman Alfitriansyah",
    "homepage": "https://github.com/Defani/Land-Cover-Classification",
    "license": "MIT",
    "entry": "plugin.wasm",
    "abi": 1,
    "minRustgisVersion": "0.6.0",
    "permissions": ["edit", "files"],
    "commands": [
        {"id": "panel-state", "label": "Refresh", "help": "Read the training samples and raster layers again.", "icon": "update"},
        {"id": "new-class", "label": "New Class", "help": "Add a class to draw samples of.", "icon": "unique",
         "params": '{"name": class name, "color"?: "#rrggbb"}',
         "form": [
             {"name": "name", "label": "Class name", "kind": "text", "required": True},
             {"name": "color", "label": "Colour", "kind": "text", "help": "#rrggbb, such as #1f78b4; empty for the next colour of the palette."},
         ]},
        {"id": "add-point", "label": "Add Point", "help": "Add a point sample of the chosen class where the map was clicked.", "icon": "point_tool",
         "params": '{"panel": {"point": {"lon", "lat"}, "class"?: class value}}'},
        {"id": "add-rectangle", "label": "Add Rectangle", "help": "Add a rectangle sample of the chosen class.", "icon": "g_rect",
         "params": '{"panel": {"box": {"bbox": [west, south, east, north]}, "class"?: class value}}'},
        {"id": "add-vertex", "label": "Add Vertex", "help": "Add a corner of a polygon sample.", "icon": "vertices",
         "params": '{"panel": {"vertex": {"lon", "lat"}}}'},
        {"id": "close-polygon", "label": "Close Polygon", "help": "Make a polygon sample of the chosen class from the corners added.", "icon": "polygon_tool"},
        {"id": "run", "label": "Classify", "icon": "run", "timeout": 600,
         "help": "Desktop: classify the band rasters chosen (raster layers read from files) with the training samples and the classifier settings, then add the map and its accuracy."},
        {"id": "add-results", "label": "Add Results\nto Map", "help": "Put the classified map or composite the tools made on the map as a raster layer.", "icon": "add_data"},
    ],
    "tools": [
        {"id": "classify", "name": "Classify", "toolbox": "Land Cover", "params": CLASSIFY,
         "summary": "Supervised classification of band rasters with training samples, a training/test split and an accuracy assessment."},
        {"id": "composite", "name": "Composite Bands", "toolbox": "Land Cover", "params": COMPOSITE,
         "summary": "Stack band rasters of any cell size into one multi-band GeoTIFF."},
        {"id": "assess", "name": "Accuracy Assessment", "toolbox": "Land Cover", "params": ASSESS,
         "summary": "Compare a classified map with reference samples: confusion matrix, overall accuracy, kappa, producer's and user's accuracy."},
    ],
    "tab": {
        "label": "Land Cover",
        "groups": [
            {"label": "Samples", "items": ["panel"]},
            {"label": "Image", "items": ["tool:composite"]},
            {"label": "Classify", "items": ["tool:classify", "add-results"]},
            {"label": "Accuracy", "items": ["tool:assess"]},
        ],
    },
    "panel": {
        "title": "Land Cover",
        "icon": "classify",
        "help": "Draw training samples by class, set the classifier and see its accuracy.",
        "state": "panel-state",
        "items": [
            {"kind": "choice", "id": "class", "label": "Draw samples of", "command": "panel-state", "help": "The class the next samples get."},
            {"kind": "button", "command": "new-class"},
            {"kind": "pick", "id": "point", "label": "Point", "command": "add-point", "help": "Click, then click the map."},
            {"kind": "pick", "id": "box", "label": "Rectangle", "shape": "box", "command": "add-rectangle", "help": "Click, then drag a box on the map."},
            {"kind": "pick", "id": "vertex", "label": "Polygon Vertex", "command": "add-vertex",
             "help": "Click, then click a corner on the map; again for each corner, then Close Polygon. Or draw on 'Training Polygons' with Edit > Create, choosing the class's template."},
            {"kind": "button", "command": "close-polygon"},
            {"kind": "text", "id": "samples", "label": "Samples", "text": "Open the panel to count the samples."},
            {"kind": "choice", "id": "image", "label": "Image", "options": [{"id": "*", "label": "All visible raster layers"}], "default": "*",
             "help": "The band rasters Classify reads (desktop: raster layers read from files). In the web app, run Land Cover > Classify and pick the files."},
            {"kind": "choice", "id": "grid", "label": "Grid", "default": "Finest raster",
             "options": [{"id": "Finest raster", "label": "Finest raster"}, {"id": "Coarsest raster", "label": "Coarsest raster"}, {"id": "First raster", "label": "First raster"}]},
            {"kind": "choice", "id": "algorithm", "label": "Algorithm", "command": "panel-state", "default": "Random Forest",
             "options": [{"id": a, "label": a} for a in ALGORITHMS]},
            {"kind": "slider", "id": "split", "label": "Training share (%)", "min": 10, "max": 100, "default": 70, "steps": True,
             "help": "70 = 70% of each class's sample pixels train, 30% test. 100 trains on all and reports the training accuracy only."},
            {"kind": "choice", "id": "split-by", "label": "Split by", "default": "Pixel",
             "options": [{"id": "Pixel", "label": "Pixel"}, {"id": "Sample (whole polygon)", "label": "Sample (whole polygon)"}]},
            {"kind": "slider", "id": "trees", "label": "Number of trees", "min": 1, "max": 500, "default": 100, "steps": True},
            {"kind": "slider", "id": "variables", "label": "Variables per split (0 = square root of bands)", "min": 0, "max": 24, "default": 0, "steps": True},
            {"kind": "slider", "id": "min-leaf", "label": "Min leaf population", "min": 1, "max": 50, "default": 1, "steps": True},
            {"kind": "slider", "id": "max-depth", "label": "Max depth (0 = unlimited)", "min": 0, "max": 50, "default": 0, "steps": True},
            {"kind": "choice", "id": "kernel", "label": "SVM kernel", "default": "RBF", "options": [{"id": "RBF", "label": "RBF"}, {"id": "Linear", "label": "Linear"}]},
            {"kind": "choice", "id": "cost", "label": "SVM cost (C)", "default": "10", "options": [{"id": c, "label": c} for c in ["0.1", "1", "10", "100", "1000"]]},
            {"kind": "choice", "id": "gamma", "label": "SVM gamma", "default": "0",
             "options": [{"id": "0", "label": "Auto (1 / bands)"}] + [{"id": g, "label": g} for g in ["0.01", "0.05", "0.1", "0.5", "1", "2"]]},
            {"kind": "slider", "id": "neighbours", "label": "k (neighbours)", "min": 1, "max": 50, "default": 5, "steps": True},
            {"kind": "choice", "id": "max-pixels", "label": "Max pixels per sample", "default": "500",
             "options": [{"id": n, "label": n} for n in ["100", "250", "500", "1000", "2500", "5000"]]},
            {"kind": "slider", "id": "seed", "label": "Seed", "min": 0, "max": 1000, "default": 42, "steps": True},
            {"kind": "button", "command": "run"},
            {"kind": "button", "command": "add-results"},
            {"kind": "text", "id": "report", "label": "Accuracy", "text": "Run Classify to see the accuracy here."},
        ],
    },
}

ID = {
    "Land Cover Classification": "Klasifikasi Tutupan Lahan",
    DESCRIPTION: (
        "Klasifikasi tutupan lahan terbimbing seperti di Google Earth Engine: gambar sampel training di peta berupa titik, "
        "persegi atau poligon per kelas, gabungkan file band dengan resolusi berbeda (Composite Bands), klasifikasikan dengan "
        "Random Forest, SVM, CART, k-NN, naive Bayes, minimum distance atau maximum likelihood dengan rasio split "
        "training/test dan parameter sendiri, lalu uji hasilnya dengan confusion matrix, overall accuracy, kappa, "
        "producer's dan user's accuracy, F1 serta luas tiap kelas."
    ),
    "Land Cover": "Tutupan Lahan",
    "Samples": "Sampel",
    "Image": "Citra",
    "Classify": "Klasifikasi",
    "Accuracy": "Akurasi",
    "Classes": "Kelas",
    "Draw samples of": "Gambar sampel kelas",
    "The class the next samples get.": "Kelas untuk sampel berikutnya.",
    "Point": "Titik",
    "Rectangle": "Persegi",
    "Polygon Vertex": "Titik Sudut Poligon",
    "Close Polygon": "Tutup Poligon",
    "Click, then click the map.": "Klik, lalu klik peta.",
    "Click, then drag a box on the map.": "Klik, lalu seret kotak di peta.",
    "Classifier": "Pengklasifikasi",
    "Add Results\nto Map": "Tambah Hasil\nke Peta",
    "Refresh": "Muat Ulang",
    "Apply\nClasses": "Terapkan\nKelas",
    "Add Point": "Tambah Titik",
    "Add Rectangle": "Tambah Persegi",
    "Add Vertex": "Tambah Titik Sudut",
    "Composite Bands": "Gabung Band (Komposit)",
    "Accuracy Assessment": "Uji Akurasi",
    "Open the panel to count the samples.": "Buka panel untuk menghitung sampel.",
    "Run Classify to see the accuracy here.": "Jalankan Klasifikasi untuk melihat akurasinya di sini.",
    "Draw training samples by class, set the classifier and see its accuracy.": "Gambar sampel training per kelas, atur pengklasifikasi dan lihat akurasinya.",
    "Algorithm": "Algoritma",
    "Training share": "Porsi training",
    "Split by": "Split berdasarkan",
    "Number of trees": "Jumlah pohon (ntrees)",
    "Variables per split": "Variabel per split",
    "Min leaf population": "Populasi daun minimum",
    "Max depth": "Kedalaman maksimum",
    "Max pixels per sample": "Piksel maksimum per sampel",
    "Output name": "Nama keluaran",
    "Training points": "Titik training",
    "Training polygons": "Poligon training",
    "Class field": "Field kelas",
    "Classified raster": "Raster hasil klasifikasi",
    "Reference samples": "Sampel referensi",
    "Finest raster": "Raster terhalus",
    "Coarsest raster": "Raster terkasar",
    "First raster": "Raster pertama",
    "Pixel": "Piksel",
    "Sample (whole polygon)": "Sampel (poligon utuh)",
}
MANIFEST["i18n"] = {"id": ID}

if __name__ == "__main__":
    (ROOT / "plugin.json").write_text(json.dumps(MANIFEST, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"plugin.json: {len(CLASSIFY)}, {len(COMPOSITE)} and {len(ASSESS)} tool parameters")
