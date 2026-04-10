# Generates two Excels files:
#   - a "most_cited.xlsx" file with the most cited papers in the dataset
#       (works that amass 80% of total citations)
#   - a "most_recent.xlsx" file with the most recent papers in the dataset
#       (within the works amassing the remaining 20% of total citations)
# Input:
#   - a CSV file with a Scopus export of papers
#   - a list of papers ordered by number of citations

import numpy as np
import pandas as pd


def main():

    # Load the dataset
    df = pd.read_csv("literature_review/scopus_export.csv").set_index("Number")

    # Load the list of most cited papers
    with open("literature_review/papers.txt", "r") as f:
        lines = f.read().splitlines()
    # Lines are given as "Paper number (citations)"
    # Calculates the number of papers that amass 80% of total citations
    # and separates them from the remaining papers
    ids = [int(line.split(" ")[0]) for line in lines]
    cumsum = np.cumsum(
        [
            int(line.split(" ")[1].strip("():"))
            for line in lines
        ]
    )
    total_citations = cumsum[-1]
    pareto_threshold = total_citations * 0.8
    works_in_pareto = next(
        i
        for i, cumsum in enumerate(cumsum)
        if cumsum >= pareto_threshold
    )

    # Get the top papers (80% of total citations)
    most_cited_ids = ids[:works_in_pareto]
    # Get the remaining papers (20% of total citations)
    remaining = ids[works_in_pareto:]

    # Filter the dataset to get the most cited papers
    most_cited_df = df.loc[most_cited_ids].sort_values(
        by="Year", ascending=False)

    # Get the remaining papers
    remaining_df = df.loc[remaining]

    # Sort the remaining papers by publication year and get the most recent
    most_recent_df = remaining_df.sort_values(by="Year", ascending=False)

    # Save the results to Excel files
    most_cited_df.to_excel("literature_review/most_cited.xlsx", index=False)
    most_recent_df.to_excel("literature_review/most_recent.xlsx", index=False)


if __name__ == "__main__":
    main()
