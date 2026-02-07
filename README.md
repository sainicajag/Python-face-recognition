# Mini face recognition project

This project takes two images, uses face recognition and OpenCV to decide whether a person appears in an image or not.

**Link to project:** 

## How it is made:

**Tech used:** Python, OpenCV, face_recognition, scikit-learn

**Face recognition** <br>
Uses OpenCV tool Haar Cascade to find general regions in which a face may appear. This is applied to the first image to detect multiple face candidates, which are stored in the "stored-faces" folder. For each candidate, a cropped image is created by extracting the bounding box region and saving it as a separate file.

**Embeddings of seen faces + storing, Embeddings of unseen face** <br>
A second detection step is then performed using face_recognition (dlib). Each cropped candidate is verified to ensure it contains a valid face, after which face embeddings are generated.
If no embedding is produced, the candidate is discarded. Otherwise, a tuple of (filename, embedding) is appended to the stored list.
The same embedding process is applied to the query image - the image being checked for presence in the original scene. If a face is successfully detected, its embedding is generated.

**Comparison** <br>
Finally, embedding comparisons are made, to see which embedding is closest to our query embedding - this will be our face match. This is done using scikit-learn’s pairwise metrics module. Cosine distance is used to identify the stored face embedding closest to the query embedding. If no valid match is found, the system returns "Face not detected (no match)". Otherwise, the cropped image of the closest match is displayed in a pop-up window.


## Optimisations

> Reduced false negatives by tuning Haar Cascade parameters (scaleFactor, minNeighbors, minSize)
> Filtered candidate regions (using Haar Cascade then face_recognition) before embedding to avoid unnecessary comparisons
> Early termination when no valid face embeddings are detected (will never be a match if there are no faces detected in initial image) 


## Lessons Learned:

Initially, I had tried to compute the embeddings with imgbeddings which was no longer compatible (written for older versions). Resolving this would require either downgrading from huggingface_hub - risking incompatability with other ML dependencies and potentially breaking tools, 
or embedding face_recognition instead, which was what was done!

During testing, the image wasn't being detected despite being a match. This prompted a systematic investigation into possible causes, including file loading, face detection or similarity calculations.
Ultimately, the root cause was traced back to having overly strict Haar Cascade parameters. By tuning haar_cascade.detectMultiScale - more specifically scaleFactor, minNeighbors, minSize - face detection performance improved (from Number of Haar-detected faces: 0, Number of stored faces with encodings: 0 to Number of Haar-detected faces: 4, Number of stored faces with encodings: 3) and valid matches were successfully identified.

