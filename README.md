# Classification Model (Paradigma Soft)


Сreation of a system for automatic receipt of a customs code based on a text description and additional descriptions and documents.


Intended applications:

· Encoder assistant. 
A module in the registry that will offer codes based on the information provided in the coding application 

· The assistant of the inspector. 
A module in the registry that tells the verifier suspicious codes when checking declarations 

· Webservice 
The client enters the required descriptions and the system offers the code

· Coding system for customs clearance of parcels from trading platforms like Ali, integration with the sites will be required here


Technologies used: 
SGDClassifier - linear classifier (SVM, logistic regression, SGD);
KNeighborsClassifier - implements k-nearest neighbor voting;
Model ensembling

Author: Arina Starceva
