import cv2, os

# ----------------- Face Recognition 
# we do this again with face_recognition - double detection

alg = "/Users/sainica/Downloads/Projects/Python Face Recognition/haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

filename = "/Users/sainica/Downloads/Projects/Python Face Recognition/gettyimages-1097661412.jpg"
img_gray = cv2.imread(filename, 0)
img_color = cv2.imread(filename)

print("Cascade loaded?", not haar_cascade.empty())
print("img_gray is None?", img_gray is None)
print("img_color is None?", img_color is None)

if img_gray is None:
    raise SystemExit("Image path is wrong OR OpenCV can't read this file.")


faces = haar_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=3, minSize=(30,30))

os.makedirs("stored-faces", exist_ok=True) #ie if it exists, that's okay - is set to True

i = 0
for x,y,w,h in faces:
    cropped_img = img_color[y:y+h, x:x+w]
    target_filename = "stored-faces/"+str(i)+".jpg"
    cv2.imwrite(target_filename,cropped_img)
    i+=1

print("Number of Haar-detected faces:", len(faces))


# ----------------- Embeddings of seen faces
    
import face_recognition

stored = []

#think of what we have stored in stored-faces as candidates, face-like regions, now we will be going through them and seeing if they are actually faces or not

for filename in os.listdir("stored-faces"):
    path = "stored-faces/" + filename

    img = face_recognition.load_image_file(path)
    encs = face_recognition.face_encodings(img)
    # list of face embeddings
    # if a face is found, atleast one embedding in encs
    # else no embeddings in encs
    
    if len(encs)==0:
        continue #ie skip eveything below if no face found 

    emb = encs[0]
    stored.append((filename,emb))


print("Number of stored faces with encodings:", len(stored))
print("Stored face filenames:", [f for f, _ in stored])

# ----------------- Embeddings of unseen face

# given that we are actually giving a face
    
query_path = "/Users/sainica/Downloads/Projects/Python Face Recognition/BTS_Jin_at_Maison_Fred,_13_March_2025_04.png"

query_img = face_recognition.load_image_file(query_path)
query_encs = face_recognition.face_encodings(query_img)

if len(query_encs)==0:
    print("No face given")
    raise SystemExit

query_emb = query_encs[0]

print("Number of faces detected in query image:", len(query_encs))

# ----------------- Comparison
from sklearn.metrics.pairwise import cosine_distances #science kit learn

best_filename = None
best_score = float("inf") #ie start with largest - infinity - and get smaller

for filename,emb in stored:
    score = cosine_distances([query_emb],[emb])[0][0]
    if score < best_score:
        best_score = score
        best_filename = filename

if best_filename is not None:
    matched = cv2.imread("stored-faces/"+best_filename)
    print("About to show best match:", best_filename)
    cv2.imshow("Best match ", matched)
    cv2.waitKey(0) #wait until the user presses a key - 0 means wait indefinitely
        # 0 - wait indefinitely until condition is met, in this case key press
        # 1 millisecond wait, then continue automatically
        # 5 milliseconds wait, then continue automatically
        # etc
    cv2.destroyAllWindows() #closes all opencv image windows
else:
    print("Face not detected (no match)")