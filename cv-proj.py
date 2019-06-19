import cv2
import os
import math
import numpy as np

def get_path_list(root_path):
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of the sub-directories in the
            root directory
    '''

    #root_path = 'dataset/train'
    train_names = []

    for name in os.listdir(root_path):
        train_names.append(name)
    
    return train_names
    # return Aaron_Eckhart, Aaron_Guiel, Aaron_Patterson, ...

def get_class_names(root_path, train_names):
    '''
        To get a list of train images path and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories
        
        Returns
        -------
        list
            List containing all image paths in the train directories
        list
            List containing all image classes id
    '''
    image_path_list = []
    image_classes_list = []

    for class_id, name in enumerate(os.listdir(root_path)):
        image_class_path = root_path + '/' + name
        # dataset/train/Aaron_Eckhart

        for image in os.listdir(image_class_path):
            image_path_list.append(root_path + '/' + name + '/' + image)
            image_classes_list.append(class_id)
            # dataset/train/Aaron_Eckhart/Aaron_Eckhart_0001.jpg

    return image_path_list, image_classes_list

def get_train_images_data(image_path_list):
    '''
        To load a list of train images from given path list

        Parameters
        ----------
        image_path_list : list
            List containing all image paths in the train directories
        
        Returns
        -------
        list
            List containing all loaded train images
    '''

    train_image_list = []

    for imagepath in image_path_list:
        train_image_list.append(cv2.imread(imagepath)) 

    return train_image_list

def detect_faces_and_filter(image_list, image_classes_list=None):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is more or less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list, optional
            List containing all image classes id
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
        list
            List containing all filtered image classes id
    '''
    train_face_grays = []
    test_faces_rects = []
    filtered_classes_list = []

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    for i, image in enumerate(image_list):
        temp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(temp, scaleFactor=1.2, minNeighbors=5)

        if (len(detected_faces) != 1) :
            continue
            
        for face in detected_faces:
            x, y, w, h = face

            face_gray_rect = temp[y:y+h , x:x+w]
            train_face_grays.append(face_gray_rect)

            test_faces_rects.append(face)

            if image_classes_list is not None:
                
                filtered_classes_list.append(image_classes_list[i])
        
            else:
                filtered_classes_list.append([''])


    return train_face_grays, test_faces_rects, filtered_classes_list

def train(train_face_grays, image_classes_list):
    '''
        To create and train classifier object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id
        
        Returns
        -------
        object
            Classifier object after being trained with cropped face images
    '''
    face_detect_object = cv2.face.LBPHFaceRecognizer_create()

    face_detect_object.train(train_face_grays, np.array(image_classes_list))

    return face_detect_object

def get_test_images_data(test_root_path, image_path_list):
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory
        image_path_list : list
            List containing all image paths in the test directories
        
        Returns
        -------
        list
            List containing all loaded test images
    '''
    test_image_list = []

    for image in os.listdir(test_root_path):
        
        test_image_list.append(cv2.imread(test_root_path+'/'+image))

    return test_image_list

def predict(classifier, test_faces_gray):
    '''
        To predict the test image with classifier

        Parameters
        ----------
        classifier : object
            Classifier object after being trained with cropped face images
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''
    result_list = []

    for image in test_faces_gray:
        result, _ = classifier.predict(image)

        result_list.append(result)

    return result_list


def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):
    '''
        To draw prediction results on the given test images

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn with
            prediction result
    ''' 

    drawn_list = []

    for i,face in enumerate(test_faces_rects):

        x, y, w, h = face

        cv2.rectangle(test_image_list[i], (x,y), (x+w,y+h), (0,255,0), 2)

        newimg = cv2.putText(test_image_list[i], train_names[predict_results[i]], (x,y-10), 1, 1, (0,255,0))

        drawn_list.append(newimg)
    
    return drawn_list


def combine_results(predicted_test_image_list):
    '''
        To combine all predicted test image result into one image

        Parameters
        ----------
        predicted_test_image_list : list
            List containing all test images after being drawn with
            prediction result

        Returns
        -------
        ndarray
            Array containing image data after being combined
    '''
    
    images = predicted_test_image_list
    
    final_image_result = images[0]

    for i in range(1, len(images)):
        final_image_result = np.hstack((final_image_result, images[i]))

    return final_image_result

def show_result(image):
    '''
        To show the given image

        Parameters
        ----------
        image : ndarray
            Array containing image data
    '''

    cv2.imshow("results",image)
    cv2.waitKey(0)

'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''
if __name__ == "__main__":
    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = "dataset/train"
    '''
        -------------------
        End of modifiable
        -------------------
    '''
    
    train_names = get_path_list(train_root_path)
    image_path_list, image_classes_list = get_class_names(train_root_path, train_names)
    train_image_list = get_train_images_data(image_path_list)
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    classifier = train(train_face_grays, filtered_classes_list)

    '''
        Please modify test_image_path value according to the location of
        your data test root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = "dataset/test"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    test_names = get_path_list(test_root_path)
    test_image_list = get_test_images_data(test_root_path, test_names)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    predict_results = predict(classifier, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names)
    final_image_result = combine_results(predicted_test_image_list)
    show_result(final_image_result)