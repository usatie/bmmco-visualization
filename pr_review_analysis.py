import requests
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Replace these with your own details
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')  # Use environment variable for security
REPO_OWNER = 'CSC-648-SFSU'
REPO_NAME = 'csc648-fa24-03-team01'

# Headers for authentication
headers = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github+json'
}

def get_all_prs():
    prs = []
    page = 1
    per_page = 100
    while True:
        url = f'https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls'
        params = {'state': 'all', 'per_page': per_page, 'page': page}
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Error fetching PRs: {response.json()}")
            break
        data = response.json()
        if not data:
            break
        prs.extend(data)
        page += 1
    return prs

def get_reviews_for_pr(pr_number):
    reviews = []
    page = 1
    per_page = 100
    while True:
        url = f'https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}/reviews'
        params = {'per_page': per_page, 'page': page}
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        if not data:
            break
        reviews.extend(data)
        page += 1
    return reviews

def analyze_reviews(prs):
    analysis_data = []
    for pr in prs:
        pr_number = pr['number']
        pr_created_at = datetime.strptime(pr['created_at'], '%Y-%m-%dT%H:%M:%SZ')
        reviews = get_reviews_for_pr(pr_number)
        reviews_sorted = sorted(reviews, key=lambda x: x['submitted_at'])
        for idx, review in enumerate(reviews_sorted):
            review_submitted_at = datetime.strptime(review['submitted_at'], '%Y-%m-%dT%H:%M:%SZ')
            time_to_review = (review_submitted_at - pr_created_at).total_seconds() / 3600  # in hours
            analysis_data.append({
                'PR Number': pr_number,
                'Review Number': idx + 1,
                'Time to Review (hours)': time_to_review
            })
    return analysis_data

def analyze_time_to_2nd_review(prs):
    analysis_data = []
    for pr in prs:
        pr_number = pr['number']
        pr_title = pr['title']
        pr_created_at = datetime.strptime(pr['created_at'], '%Y-%m-%dT%H:%M:%SZ')
        reviews = get_reviews_for_pr(pr_number)
        reviews_sorted = sorted(reviews, key=lambda x: x['submitted_at'])

        if len(reviews_sorted) >= 2:
            second_review = reviews_sorted[1]
            second_review_submitted_at = datetime.strptime(second_review['submitted_at'], '%Y-%m-%dT%H:%M:%SZ')
            time_to_second = (second_review_submitted_at - pr_created_at).total_seconds() / 3600  # in hours
            analysis_data.append({
                'PR Number': pr_number,
                'PR Title': pr_title,
                'PR Created At': pr_created_at,
                'Time to Second Review (hours)': time_to_second
            })
    return analysis_data

def visualize_data(times, analysis_data):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    times_series = pd.Series(times)
    pr_created_at = pd.Series([item['PR Created At'] for item in analysis_data])

    sns.set(style='whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    sns.histplot(times_series, bins=20, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Distribution of Time to Second Review')
    axes[0].set_xlabel('Time to Second Review (hours)')
    axes[0].set_ylabel('Frequency')

    # Prepare data for Bar Plot
    df_plot = pd.DataFrame({
        'PR Created At': pr_created_at,
        'Time to Second Review (hours)': times_series
    })
    df_plot.sort_values('PR Created At', inplace=True)

    # Group by day
    df_plot['Day'] = df_plot['PR Created At'].dt.date
    grouped = df_plot.groupby('Day')['Time to Second Review (hours)'].mean().reset_index()

    # Bar Plot
    sns.barplot(data=grouped, x='Day', y='Time to Second Review (hours)', ax=axes[1], color='lightblue')
    axes[1].set_title('Average Time to Second Review per Day')
    axes[1].set_xlabel('Day')
    axes[1].set_ylabel('Time to Second Review (hours)')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('time_to_second_review_plots.png')
    plt.show()




def main():
    print("Fetching all PRs...")
    prs = get_all_prs()
    print(f"Total PRs fetched: {len(prs)}")
    
    # Filter out PRs that are not merged nor open
    prs = [pr for pr in prs if pr['state'] == 'open' or pr['merged_at']]

    # Filter out old PRs (older than a week)
    prs = [pr for pr in prs if (datetime.now() - datetime.strptime(pr['created_at'], '%Y-%m-%dT%H:%M:%SZ')).days <= 7]

    print("Analyzing reviews...")
    analysis_data = analyze_reviews(prs)

    df = pd.DataFrame(analysis_data)
    print("Aggregated Data:")
    print(df)

    # Optionally, save to CSV
    df.to_csv('pr_review_analysis.csv', index=False)
    print("Data saved to pr_review_analysis.csv")

    print("Analyzing time to 2nd review...")
    time_to_second_review = analyze_time_to_2nd_review(prs)

    if time_to_second_review:
        min_time = min(item['Time to Second Review (hours)'] for item in time_to_second_review)
        max_time = max(item['Time to Second Review (hours)'] for item in time_to_second_review)
        avg_time = sum(item['Time to Second Review (hours)'] for item in time_to_second_review) / len(time_to_second_review)

        print(f"\nStatistics for Time to Second Review (in hours):")
        print(f"Minimum time: {min_time:.2f} hours")
        print(f"Maximum time: {max_time:.2f} hours")
        print(f"Average time: {avg_time:.2f} hours")
        times = [item['Time to Second Review (hours)'] for item in time_to_second_review]
        visualize_data(times, time_to_second_review)
    else:
        print("No PRs with at least two reviews were found.")

if __name__ == '__main__':
    main()

