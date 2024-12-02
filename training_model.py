import face_recognition
import os
import pickle

known_face_encodings = []
known_face_names = []

for person_name in os.listdir('dataset'):
    person_folder = os.path.join('dataset', person_name)
    if os.path.isdir(person_folder):
        for image_file in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_file)
            
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)

            if len(face_locations) > 0:
                face_encodings = face_recognition.face_encodings(image, face_locations)
                
                if len(face_encodings) > 0:
                    face_encoding = face_encodings[0]
                    
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(person_name)

with open('face_data.pkl', 'wb') as f:
    pickle.dump((known_face_encodings, known_face_names), f)

print(f"Saved {len(known_face_encodings)} face encodings to 'face_data.pkl'.")
