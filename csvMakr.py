import os
import csv

# Paths to the folders containing the reviews
positive_reviews_path = 'aclimdb/train/pos'
negative_reviews_path = 'aclimdb/train/neg'
output_csv_path = 'reviews.csv'

def read_reviews_from_folder(folder_path, sentiment_label):
    reviews = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                review_text = file.read().strip()
                reviews.append((review_text, sentiment_label))
    return reviews

def write_reviews_to_csv(reviews, csv_path):
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['review', 'sentiment'])
        csvwriter.writerows(reviews)

def main():
    positive_reviews = read_reviews_from_folder(positive_reviews_path, 'positive')
    negative_reviews = read_reviews_from_folder(negative_reviews_path, 'negative')

    all_reviews = positive_reviews + negative_reviews

    write_reviews_to_csv(all_reviews, output_csv_path)

if __name__ == "__main__":
    main()
