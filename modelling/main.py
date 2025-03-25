from modelling.Model import Model
from preprocessor.DataPreprocessor import DataPreprocessor

def main():
    # Initialize the model
    model = Model(file_path='data/train.csv', target_column='Price')

    # Load and preprocess the data
    model.load_data()
    model.preprocess_data()

    # Train the model
    model.train(learning_rate=0.001, epochs=2000)

    # Evaluate the model on the test set
    model.evaluate()

    # Plot the train-validation loss curve
    model.plot_loss_curve()

if __name__ == "__main__":
    main()