import os
import pandas as pd


def create_stratified_sample(input_file, output_file, sample_fraction=0.10, random_state=42):
    if not os.path.exists(input_file):
        print(f"ERROR: File not found: {input_file}")
        return False

    backup_file = input_file.replace('.csv', '.backup.csv')

    print(f"\nLoading dataset")

    try:
        df = pd.read_csv(input_file, low_memory=False)
        print(f"Dataset loaded")
        print(f"Shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    except Exception as e:
        print(f"Error: {e}")
        return False

    if 'loan_status' not in df.columns:
        print(f"\n'loan_status' column not found")
        df_sample = df.sample(frac=sample_fraction, random_state=random_state)
    else:
        print(f"\nCreating stratified sample ({sample_fraction*100:.0f}%)")
        print(f"Stratifying by 'loan_status'")

        print(f"\nOriginal 'loan_status' distribution:")
        status_counts = df['loan_status'].value_counts()
        for status, count in status_counts.items():
            pct = 100 * count / len(df)
            print(f"{status}: {count:,} ({pct:.2f}%)")

        try:
            df_sample = df.groupby('loan_status', group_keys=False).apply(
                lambda x: x.sample(frac=sample_fraction, random_state=random_state)
            ).reset_index(drop=True)

            print(f"\nStratified sample created")

            # Show sampled distribution
            print(f"\nSampled 'loan_status' distribution:")
            status_counts_sample = df_sample['loan_status'].value_counts()
            for status, count in status_counts_sample.items():
                pct = 100 * count / len(df_sample)
                print(f"     {status}: {count:,} ({pct:.2f}%)")

        except Exception as e:
            print(f"Error during stratified sampling: {e}")
            df_sample = df.sample(frac=sample_fraction, random_state=random_state)

    print(f"\nSaved sampled dataset")
    print(f"Output: {output_file}")
    print(f"Shape: {df_sample.shape}")

    try:
        df_sample.to_csv(output_file, index=False)
        original_size = os.path.getsize(input_file) / 1024**2
        new_size = os.path.getsize(output_file) / 1024**2
        reduction_pct = (1 - new_size/original_size) * 100
        print(f"Original: {original_size:.2f} MB")
        print(f"Sampled:  {new_size:.2f} MB")
        print(f"Reduction: {reduction_pct:.1f}%")

    except Exception as e:
        print(f"Error saving sampled dataset: {e}")
        return False

    return True


def main():
    INPUT_FILE = 'data/lending_club.csv'
    OUTPUT_FILE = 'data/lending_club_sampled.csv'
    SAMPLE_FRACTION = 0.01
    RANDOM_STATE = 42

    success = create_stratified_sample(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        sample_fraction=SAMPLE_FRACTION,
        random_state=RANDOM_STATE
    )

    if success:
        print("\nDataset sampling completed successfully!")
        return 0
    else:
        print("\nDataset sampling failed!")
        return 1


if __name__ == "__main__":
    exit(main())
