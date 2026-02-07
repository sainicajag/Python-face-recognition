# Mini face recognition project

This project takes two images, uses face recognition and OpenCV to decide whether a person appears in an image or not.

**Link to project:** 

## How it is made:

**Tech used:** Python, OpenCV, face_recognition, scikit-learn

Used OpenCV tool Haar Cascade to find general regions in which a face may appear. Applied this to the first image to find as many facial regions as possible. Stored these regions in folder "stored-faces". 
For each potential face, we take a cropped image, ie from the corner create a box, and store this in a file under the folder. 

We then do double detection using face_recognition (dlib). For the seen faces, we verify whether they are actually a face or not then create a list of face embeddings.
If this list is empty, ie we have no faces, we continue. Otherwise, we append to array stored tuple of (filename, emb), where emb is its corresponding embedding.
We do the same for our query image - ie the one we are trying to see is in the general picture. We check whether we can detect a face and if so find the embeddings.

Now, we do comparisons, to see which embedding is closest to our query embedding - this will be our face match. We do this using scikit-learn's pairwise metrics module.
We find which faces' embeddings have the smallest cosine distance between them.

If we find none, we return "Face not detected (no match)", otherwise we show the cropped_image of the match as a pop-up window. 


## Optimisations

> Reduced false negatives by tuning Haar Cascade parameters (scaleFactor, minNeighbors, minSize)
> Filtered candidate regions (using Haar Cascade then face_recognition) before embedding to avoid unnecessary comparisons
> Early termination when no valid face embeddings are detected (will never be a match if there are no faces detected in initial image) 


## Lessons Learned:

Initially, I had tried to compute the embeddings with imgbeddings which was no longer compatible (written for older versions). Resolving this would require either downgrading from huggingface_hub - risking incompatability with other ML dependencies and potentially breaking tools, 
or embedding face_recognition instead, which was what was done!

During testing, the image wasn't being detected despite being a match. This prompted a systematic investigation into possible causes, including file loading, face detection or similarity calculations.
Ultimately, the root cause was traced back to having overly strict Haar Cascade parameters. By tuning haar_cascade.detectMultiScale - more specifically scaleFactor, minNeighbors, minSize - face detection performance improved (from Number of Haar-detected faces: 0, Number of stored faces with encodings: 0 to Number of Haar-detected faces: 4, Number of stored faces with encodings: 3) and valid matches were successfully identified.

