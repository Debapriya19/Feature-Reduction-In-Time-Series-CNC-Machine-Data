import argparse
from config import Config
from data.dataloader import load_dataset, make_sequences
from models import lstm_cnn, dense_ae, pca_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=Config().data_path)
    p.add_argument("--window_size", type=int, default=Config().window_size)
    p.add_argument(
        "--split", type=float, nargs=3, default=list(Config().split),
        help="train val test fractions, e.g. 0.6 0.2 0.2"
    )
    p.add_argument("--model", type=str, required=True,
                   choices=["pca_model", "dense_ae", "lstm_cnn"])
    return p.parse_args()


def main():
    args = parse_args()
    split_tuple = tuple(args.split)

    # Load & scale (no windows here)
    X_train, X_val, X_test, y_train, y_val, y_test, x_scaler, y_scaler = load_dataset(
        path=args.data,
        split=split_tuple,
        feature_cols_slice=slice(0, 52),   # first 52 features
        target_col="CURRENT|6",
    )

    if args.model == "lstm_cnn":
        # Build sequences with the chosen window size
        X_tr_seq, y_tr_seq = make_sequences(X_train, y_train, args.window_size, predict_offset=1)
        X_va_seq, y_va_seq = make_sequences(X_val, y_val, args.window_size, predict_offset=1)
        X_te_seq, y_te_seq = make_sequences(X_test, y_test, args.window_size, predict_offset=1)

        lstm_cnn.run(X_tr_seq, X_va_seq, X_te_seq,
                     y_tr_seq, y_va_seq, y_te_seq,
                     x_scaler, y_scaler)

    elif args.model == "dense_ae":
        dense_ae.run(X_train, X_val, X_test,
                     y_train, y_val, y_test,
                     x_scaler, y_scaler)

    elif args.model == "pca_model":
        pca_model.run(X_train, X_val, X_test,
                      y_train, y_val, y_test)


if __name__ == "__main__":
    main()