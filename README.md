## Diabetic Retinopathy

Our project's main goal is to develop and implement a convolutional neural network (CNN) model that can recognize symptoms of diabetic retinopathy in retinal images with accuracy. Our goal is to train our model as effectively as possible across distributed computer resources by utilizing cutting-edge deep learning techniques like the Mirrored Strategy, thus guaranteeing scalability and performance.

Additionally, we are investigating the use of TensorFlow Serving, a versatile and scalable serving system for machine learning models, for the deployment of our trained model. We anticipate a smooth integration of our model into current workflows through deployment in real-world healthcare settings, allowing healthcare providers to diagnose patients promptly and accurately.

Our goal is to advance medical diagnostics significantly through this initiative by providing healthcare providers with AI-powered tools that improve patient care and results. Our goals are to lessen the strain on healthcare systems, increase accessibility to screening programs, and eventually stop vision loss in people with diabetes by automating the identification of diabetic retinopathy.

Docker Tensorflow Serving
To install docker

````bash
docker pull tensorflow/serving

To Serve Only Latest Model
```bash
docker run -it -v <path/to/model>:/<name>-p 8601:8601 --entrypoint /bin/bash tensorflow/serving

```bash
tensorflow_model_server --rest_api_port=8601 --model_name=reinopathy --model_base_path=/retinopathy/models/
````
