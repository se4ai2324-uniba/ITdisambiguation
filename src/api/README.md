APIs
================
This folder contains the source code for the APIs of our model.
APIs are hosted using uvicorn and the server can be launched simply by executing the following command while in the project's root folder
```
uvicorn src.api.api:app
```

Otherwise, if you want to use the APIs hosted by our server, you can connect to the following URL
```
https://itdisambiguation.azurewebsites.net/
```

Endpoints
----------------
The following endpoints are exposed by the server:
- **/models**: Gets a list of all the available models
- **/models/{model_name}**: Gets more informations about the specified model
- **/models/{model_name}/predict_context**: Predict the most relevant context given a list of contexts, an image and a target word
- **/models/{model_name}/predict_images**: Predict the most relevant image given a list of images, a context and a target word

Prometheus and Graphana
================
TODO
