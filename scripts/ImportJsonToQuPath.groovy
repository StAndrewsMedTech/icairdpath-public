import qupath.lib.io.GsonTools
import qupath.lib.objects.PathObject
import qupath.lib.geom.Point2
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.roi.PolygonROI
import qupath.lib.roi.RoiTools

jsondir = "json_files_to_import"

def imageData = QPEx.getCurrentImageData()
def server = imageData.getServer()
def filename = server.getFile().getName()
def filenamestem = filename.substring(0,14)
def inputname = filenamestem + ".json"

def path = buildFilePath(PROJECT_BASE_DIR, jsondir, inputname)


def gson=GsonTools.getInstance(true)
BufferedReader bufferedReader = new BufferedReader(new FileReader(path));
HashMap<String, String> myjson = gson.fromJson(bufferedReader, HashMap.class);


labelValues = myjson.keySet()
print(labelValues)

for (label in labelValues) {
    def annots = myjson[label]
    def pathclazz = getPathClass(label)
    annotations = []
    for (ann in annots) {
        points = []
        poly = ann[0]
        for (pts in poly) {
            points.add(new Point2(pts[0]*32, pts[1]*32))
        }
        def outer_polygon = new PolygonROI(points)
        ann.remove(0)
    
        while (ann.size > 0) {
            points = []
            in_poly = ann[0]
            for (pts in in_poly) {
                points.add(new Point2(pts[0]*32, pts[1]*32))
            }
            def hole = new PolygonROI(points)
            outer_polygon = RoiTools.combineROIs(outer_polygon, hole, RoiTools.CombineOp.SUBTRACT)
            ann.remove(0)
        }
    
        def pathAnnotation = new PathAnnotationObject(outer_polygon)
        annotations << pathAnnotation
        pathAnnotation.setPathClass(pathclazz)   
    }
}


// Add to current hierarchy
QPEx.addObjects(annotations)

