# Farm Vision ML API

The Farm Vision ML API is a Flask-based web application designed to integrate machine learning models seamlessly into the Farm Vision platform. It provides endpoints for running various models to enhance decision-making and optimize farm operations.

## Features

- **Model Execution**: Run machine learning models to provide insights and predictions for farm management tasks.
- **Integration**: Seamlessly integrates with the Farm Vision API, allowing users to access ML functionalities from within the platform.
- **Scalability**: Built on Flask, the API is scalable and can accommodate additional models as needed.
- **Security**: Utilizes secure authentication and authorization mechanisms to ensure the confidentiality of data and model results.
- **Database Interaction**: Utilizes SQLAlchemy for interaction with the database, enabling efficient data storage and retrieval.

## Technologies Used

- **Flask**: Lightweight web framework for Python used to build the API endpoints.
- **SQLAlchemy**: SQL toolkit and Object-Relational Mapping (ORM) for Python, facilitating interaction with the database.
- **Machine Learning Libraries**: Various Python libraries such as TensorFlow for implementing machine learning models.

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/Mohamed-Twfik/Farm-Vision-ML-API.git
   ```

2. Install dependencies:

   ```bash
   cd Farm-Vision-ML-API
   pip install -r requirements.txt
   ```

3. Set up environment variables:

   Create a `.env` file in the root directory and add the following variables:

   ```plaintext
   DATABASE_URL= your Database URL
   ```

4. Start the Flask server:

   ```bash
   python app.py
   ```

5. Access the API endpoints at `http://localhost:5000`.

<!-- ## API Documentation

- Detailed API documentation can be found in the [API Documentation](./API_DOCUMENTATION.md) file. -->

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Create a new Pull Request.


## License

This project is licensed under the [MIT License](./LICENSE).