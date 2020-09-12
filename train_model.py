# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle


#path to serialized db of facial embeddings
embeddings ='./output/embeddings.pickle'

#path to output model trained to recognize faces
recognizer_path = './output/recognizer.pickle'

#path to output label encoder
le_path = './output/le.pickle'




# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(embeddings, "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
f = open(recognizer_path, "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open(le_path, "wb")
f.write(pickle.dumps(le))
f.close()