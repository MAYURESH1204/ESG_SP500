from flask import Flask, render_template, request, send_from_directory
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Flask app setup
app = Flask(__name__)

# Set a good visual theme
sns.set(style="whitegrid")

@app.route('/', methods=['GET', 'POST'])
def index():
    scatter_plot = None
    box_plot = None
    csv_file = None
    error = None

    if request.method == 'POST':
        try:
            # Your analysis code here, e.g. fetching data, calculating, plotting

            # For example:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            sp500_table = pd.read_html(url)
            sp500 = sp500_table[0]
            sp500 = sp500[['Symbol', 'Security', 'GICS Sector']]
            sp500.columns = ['Ticker', 'Company', 'Sector']

            # Simulated ESG Scores
            np.random.seed(42)
            sp500['E_Score'] = np.random.randint(50, 100, size=len(sp500))
            sp500['S_Score'] = np.random.randint(50, 100, size=len(sp500))
            sp500['G_Score'] = np.random.randint(50, 100, size=len(sp500))
            sp500['ESG_Score'] = sp500[['E_Score', 'S_Score', 'G_Score']].mean(axis=1)

            def get_stock_return(ticker):
                ticker = ticker.replace('.', '-')
                try:
                    data = yf.download(ticker, period="1y", progress=False)
                    time.sleep(1)
                    if data.empty:
                        return np.nan
                    start_price = data['Close'].iloc[0]
                    end_price = data['Close'].iloc[-1]
                    return float((end_price - start_price) / start_price * 100)
                except Exception as e:
                    print(f"Error fetching {ticker}: {e}")
                    return np.nan

            subset = sp500.head(50).copy()
            subset['1Y_Return'] = subset['Ticker'].apply(get_stock_return)
            subset = subset.dropna(subset=['1Y_Return'])

            # Correlation print (optional)
            correlation = subset[['ESG_Score', '1Y_Return']].corr()
            print(correlation)

            # Scatter Plot
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=subset, x='ESG_Score', y='1Y_Return', hue='Sector')
            plt.title("ESG Score vs 1-Year Stock Return")
            plt.xlabel("ESG Score")
            plt.ylabel("1-Year Return (%)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            scatter_path = "static/scatter.png"
            plt.savefig(scatter_path, bbox_inches='tight')
            plt.close()

            # Box Plot
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=subset, x='Sector', y='ESG_Score')
            plt.xticks(rotation=90)
            plt.title("Sector-wise ESG Score Distribution")
            box_path = "static/boxplot.png"
            plt.savefig(box_path, bbox_inches='tight')
            plt.close()

            # Save CSV
            csv_file = "static/esg_sp500_analysis.csv"
            subset.to_csv(csv_file, index=False)

            return render_template('index.html',
                                   scatter_plot=scatter_path,
                                   box_plot=box_path,
                                   csv_file=csv_file)

        except Exception as e:
            error = f"Error occurred: {str(e)}"
            return render_template('index.html', error=error)

    # For GET requests, just render the page without plots or CSV
    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory('static', filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
