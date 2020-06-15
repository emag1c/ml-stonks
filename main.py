import tensorflow as tf
import os
import pandas as pd
# from sklearn.preprocessing import
tf.random.set_seed(123)

if __name__ == '__main__':
    # train in QQQ
    train = pd.read_csv(os.path.join(os.getcwd(), "data", "QQQ_5y_train.csv"), header=0)  # type: pd.DataFrame

    ver = pd.read_csv(os.path.join(os.getcwd(), "data", "SPY_5y_train.csv"), header=0)  # type: pd.DataFrame

    # train
    y_train = train.pop("%change-6")
    # verify
    y_ver = ver.pop("%change-6")

    print(train)
    print(ver)

    NUMERIC_COLUMNS = ver.columns.to_list()

    feature_columns = []

    # add the feature columns
    for feature_name in NUMERIC_COLUMNS:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.dtypes.float64))

    NUM_EXAMPLES = len(y_train)

    def make_input_fn(X, y, n_epochs=None, shuffle=True):
        def input_fn():
            dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
            if shuffle:
                dataset = dataset.shuffle(NUM_EXAMPLES)
            # for training, cycle through the dataset
            dataset = dataset.repeat(n_epochs)
            return dataset.batch(NUM_EXAMPLES)
        return input_fn

    train_input_fn = make_input_fn(train, y_train)
    eval_input_fn = make_input_fn(ver, y_ver, shuffle=False, n_epochs=1)

    # LINEAR BENCHMARK
    linear_est = tf.estimator.LinearClassifier(feature_columns)

    # Train
    linear_est.train(train_input_fn, max_steps=100)

    # Evaluation
    result = linear_est.evaluate(eval_input_fn)
    print(pd.Series(result))

