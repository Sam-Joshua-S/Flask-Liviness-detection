# Human-Authentication-using-Facial-and-Palm-Recognition

The main aim of our project is to introduce an alternative approach to passwords and captcha by using face and palm recognition.
To Overcome the problem faced in existing authentication system we have introduced a new authentication system by combining the solution of facial and palm verification.

There are three main modules present in our system such as face recognition palm recognition and liveness detection.

For face recogntion and palm recognition we have used siamese network deep learning algorithm. How this algorithm works is it first gets the image from the user and then it converts the image into embeddings. then it calculates the euclidean distance between the captured image and the image which is stored in db during signup. If both the embeddings matches and the threshold is within the range the user is logged in otherwise not.

The main reason for implementing liveness detection is there is a chances of login into the system by an unauthorized person using the photograph of the authorized person. To avoid this kind of practices we have introduced liveness detection. This is implemented using efficient net pretained deep learning model. We have fine tuned this model with our own dataset images.The reason for using this model is it increases the resolution of images which will be useful for live tracking of facial movements and therefore it increases the accuracy.

After the successfull verfication of both face and palm recogntion the useer is logged in to the system.
