from datasets import load_dataset

DATASET_NAME = "rajpurkar/squad"
SPLIT = "train[:100]"


def get_data():
    print("Fetching dataset from Hugging Face Hub...")
    # using the verified repo id for SQuAD
    ds = load_dataset(DATASET_NAME, split=SPLIT)
    return ds


def main():
    try:
        ds = get_data()
        print("\nData loaded successfully!")
        print(f"Available columns: {ds.column_names}")
        print(f"First context preview: {ds[0]['context'][:200]}...")
    except Exception as err:
        print(f"\nSomething went wrong: {err}")


if __name__ == "__main__":
    main()
