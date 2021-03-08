#%%
from flask import Flask, jsonify, request, send_file, url_for
# from requests_toolbelt import MultipartEncoder
import numpy
import cv2 
import matplotlib.pyplot as plt
from imageai.Detection import ObjectDetection
import os
app = Flask(__name__)
filename = "img.bmp" 
@app.route('/getImageDetails', methods = ['POST'])
def getImageDetails():
    image = request.files
    print()
    print(image)
    filestr = image['img'].read()
    #convert string data to numpy array
    npimg = numpy.fromstring(filestr, numpy.uint8)
    # convert numpy array to image
    print(1)
    img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
    print(1)
    # img = cv2.imread(filename)
    cv2.imshow("image from post",img)
    cv2.imwrite("image_from_post_request.jpeg", img)
    print(1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # plt.imshow(img)
    # plt.show()
    filename = "image_from_post_request.jpeg"
    execution_path = os.getcwd()
    file_name = "image_from_post_request"
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , file_name + ".jpeg"), output_image_path=os.path.join(execution_path , file_name + "detected.jpeg"))
    image = cv2.imread(file_name+".jpeg")
    copy = image.copy()
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # ROI_number = 0
    # return(jsonify({"out" : "Success !!!!"}))
    output = {"humna_face" : {"count" : 0}, "animal" : {"count" : 0, "objects_found" : dict()}, "object" : {"count" : 0, "objects_found" : dict()}}
    animal = ["bird",   "cat",   "dog",   'horse',   "sheep",   "cow",   "elephant",   "bear",   "zebra", "giraffe"]
    for eachObject in detections:
        print(output, eachObject.keys())
        print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
        if(eachObject['name'] == 'person'):
            x,y,w,h = eachObject['box_points']
            ROI = image[y:y+h, x:x+w]
            cv2.imwrite('ROI.jpeg', ROI)
            cv2.rectangle(copy,(x,y),(x+w,y+h),(36,255,12),2)
            person = cv2.imread('ROI.jpeg')
            # Convert into grayscale
            gray = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.4, 1, minSize = (1, 1))
            if(len(faces) == 0):
                pass
            else:
                output['human_face']['count'] += 1
        elif(eachObject['name'] in animal):
            output['animal']['count'] += 1
            if(eachObject['name'] not in output['animal']['objects_found'].keys()):
                output['animal']['objects_found'][eachObject['name']] = 0
            output['animal']['objects_found'][eachObject['name']] += 1
    
    
        else:
            output['object']['count'] += 1
            if(eachObject['name'] not in output['object']['objects_found'].keys()):
                output['object']['objects_found'][eachObject['name']] = 0
            output['object']['objects_found'][eachObject['name']] += 1



    # return(send_file(filename_or_fp = file_name + "detected.jpeg", mimetype="image/gif"))
    # return(m.to_string(), {'Content-Type': m.content_type})
    return(jsonify(output))

if __name__ == "__main__":
    # getImageDetails()
    app.run(debug = True)


# %%
