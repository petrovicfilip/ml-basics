import logging
from stream import process_csv

csv_files = [
    r"C:\Praksa\Praksa Zad\datasets\archive\02-14-2018.csv",
    r"C:\Praksa\Praksa Zad\datasets\archive\02-15-2018.csv",
    r"C:\Praksa\Praksa Zad\datasets\archive\02-16-2018.csv",
    r"C:\Praksa\Praksa Zad\datasets\archive\02-20-2018.csv",
    r"C:\Praksa\Praksa Zad\datasets\archive\02-21-2018.csv",
    r"C:\Praksa\Praksa Zad\datasets\archive\02-22-2018.csv",
    r"C:\Praksa\Praksa Zad\datasets\archive\02-23-2018.csv",
    r"C:\Praksa\Praksa Zad\datasets\archive\02-28-2018.csv",
    r"C:\Praksa\Praksa Zad\datasets\archive\03-01-2018.csv",
    r"C:\Praksa\Praksa Zad\datasets\archive\03-02-2018.csv",
]

logging.basicConfig(level=logging.INFO)


logging.basicConfig(level=logging.INFO)


def main():
    for file_name in csv_files:
        logging.info(f"Processing file: {file_name}")
        result = process_csv(file_name)  # 
        if isinstance(result, Exception):
            logging.error(f"Error processing {file_name}: {result}")


if __name__ == "__main__":
    main()