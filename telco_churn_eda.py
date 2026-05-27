"""
Exploratory Data Analysis for Telco Customer Churn
Comprehensive analysis of customer characteristics and churn patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ChurnEDA:
    """Class for exploratory data analysis"""
    
    def __init__(self, data_path):
        """Initialize with data path"""
        self.data_path = data_path
        self.df = None
        self.churn_rate = None
        
    def load_data(self):
        """Load dataset"""
        print("Loading data for EDA...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        return self.df
    
    def basic_statistics(self):
        """Display basic statistics"""
        print("\n" + "="*60)
        print("BASIC STATISTICS")
        print("="*60)
        
        print(f"\nDataset Info:")
        print(f"Total Records: {len(self.df)}")
        print(f"Total Features: {len(self.df.columns)}")
        print(f"\nColumn Names:\n{self.df.columns.tolist()}")
        
        print(f"\nData Types:\n{self.df.dtypes}")
        print(f"\nMissing Values:\n{self.df.isnull().sum()}")
        
        # Churn statistics
        if 'Churn' in self.df.columns:
            churn_counts = self.df['Churn'].value_counts()
            self.churn_rate = churn_counts['Yes'] / len(self.df) * 100
            
            print(f"\nChurn Distribution:")
            print(f"No Churn: {churn_counts['No']} ({churn_counts['No']/len(self.df)*100:.2f}%)")
            print(f"Churned: {churn_counts['Yes']} ({churn_counts['Yes']/len(self.df)*100:.2f}%)")
    
    def numerical_analysis(self):
        """Analyze numerical features"""
        print("\n" + "="*60)
        print("NUMERICAL FEATURES ANALYSIS")
        print("="*60)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        print(f"\nNumerical columns: {numerical_cols.tolist()}")
        print(f"\nDescriptive Statistics:\n{self.df[numerical_cols].describe()}")
    
    def categorical_analysis(self):
        """Analyze categorical features"""
        print("\n" + "="*60)
        print("CATEGORICAL FEATURES ANALYSIS")
        print("="*60)
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != 'customerID':
                print(f"\n{col}:")
                print(self.df[col].value_counts())
                print(f"Unique values: {self.df[col].nunique()}")
    
    def churn_by_tenure(self):
        """Analyze churn by tenure"""
        print("\n" + "="*60)
        print("CHURN BY TENURE ANALYSIS")
        print("="*60)
        
        if 'tenure' in self.df.columns and 'Churn' in self.df.columns:
            churn_by_tenure = self.df.groupby('tenure')['Churn'].apply(
                lambda x: (x == 'Yes').sum() / len(x) * 100
            )
            print(f"\nAverage churn rate by tenure:\n{churn_by_tenure.describe()}")
    
    def churn_by_contract(self):
        """Analyze churn by contract type"""
        print("\n" + "="*60)
        print("CHURN BY CONTRACT TYPE")
        print("="*60)
        
        if 'Contract' in self.df.columns and 'Churn' in self.df.columns:
            contract_churn = pd.crosstab(
                self.df['Contract'], 
                self.df['Churn'], 
                margins=True,
                normalize='index'
            ) * 100
            print(f"\nChurn rate by contract type:\n{contract_churn}")
    
    def churn_by_internet_service(self):
        """Analyze churn by internet service type"""
        print("\n" + "="*60)
        print("CHURN BY INTERNET SERVICE TYPE")
        print("="*60)
        
        if 'InternetService' in self.df.columns and 'Churn' in self.df.columns:
            internet_churn = pd.crosstab(
                self.df['InternetService'], 
                self.df['Churn'], 
                margins=True,
                normalize='index'
            ) * 100
            print(f"\nChurn rate by internet service:\n{internet_churn}")
    
    def correlation_analysis(self):
        """Analyze correlations with churn"""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        # Create a copy and encode churn
        df_corr = self.df.copy()
        df_corr['Churn'] = (df_corr['Churn'] == 'Yes').astype(int)
        
        # Encode other categorical variables
        categorical_cols = df_corr.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'customerID':
                df_corr[col] = pd.factorize(df_corr[col])[0]
        
        # Get correlations with churn
        correlations = df_corr.corr()['Churn'].sort_values(ascending=False)
        print(f"\nCorrelation with Churn:\n{correlations}")
        
        return correlations
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Churn Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        churn_counts = self.df['Churn'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        ax1.pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax1.set_title('Churn Distribution', fontsize=12, fontweight='bold')
        
        # 2. Churn by Tenure
        ax2 = fig.add_subplot(gs[0, 1])
        churn_by_tenure = self.df.groupby('tenure')['Churn'].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100
        )
        ax2.plot(churn_by_tenure.index, churn_by_tenure.values, linewidth=2, color='#e74c3c')
        ax2.fill_between(churn_by_tenure.index, churn_by_tenure.values, alpha=0.3, color='#e74c3c')
        ax2.set_xlabel('Tenure (months)')
        ax2.set_ylabel('Churn Rate (%)')
        ax2.set_title('Churn Rate by Tenure', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Monthly Charges by Churn
        ax3 = fig.add_subplot(gs[0, 2])
        self.df.boxplot(column='MonthlyCharges', by='Churn', ax=ax3)
        ax3.set_title('Monthly Charges by Churn Status', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Churn Status')
        ax3.set_ylabel('Monthly Charges ($)')
        plt.sca(ax3)
        plt.xticks([1, 2], ['No', 'Yes'])
        
        # 4. Contract Type vs Churn
        ax4 = fig.add_subplot(gs[1, 0])
        contract_churn = pd.crosstab(self.df['Contract'], self.df['Churn'])
        contract_churn.plot(kind='bar', ax=ax4, color=['#2ecc71', '#e74c3c'])
        ax4.set_title('Churn by Contract Type', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Contract Type')
        ax4.set_ylabel('Count')
        ax4.legend(title='Churn', labels=['No', 'Yes'])
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        # 5. Internet Service vs Churn
        ax5 = fig.add_subplot(gs[1, 1])
        internet_churn = pd.crosstab(self.df['InternetService'], self.df['Churn'])
        internet_churn.plot(kind='bar', ax=ax5, color=['#2ecc71', '#e74c3c'])
        ax5.set_title('Churn by Internet Service Type', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Internet Service')
        ax5.set_ylabel('Count')
        ax5.legend(title='Churn', labels=['No', 'Yes'])
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
        
        # 6. Total Charges by Churn
        ax6 = fig.add_subplot(gs[1, 2])
        self.df.boxplot(column='TotalCharges', by='Churn', ax=ax6)
        ax6.set_title('Total Charges by Churn Status', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Churn Status')
        ax6.set_ylabel('Total Charges ($)')
        plt.sca(ax6)
        plt.xticks([1, 2], ['No', 'Yes'])
        
        # 7. Gender vs Churn
        ax7 = fig.add_subplot(gs[2, 0])
        gender_churn = pd.crosstab(self.df['gender'], self.df['Churn'])
        gender_churn.plot(kind='bar', ax=ax7, color=['#2ecc71', '#e74c3c'])
        ax7.set_title('Churn by Gender', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Gender')
        ax7.set_ylabel('Count')
        ax7.legend(title='Churn', labels=['No', 'Yes'])
        plt.setp(ax7.xaxis.get_majorticklabels(), rotation=0)
        
        # 8. Senior Citizen vs Churn
        ax8 = fig.add_subplot(gs[2, 1])
        senior_churn = pd.crosstab(self.df['SeniorCitizen'], self.df['Churn'])
        senior_churn.plot(kind='bar', ax=ax8, color=['#2ecc71', '#e74c3c'])
        ax8.set_title('Churn by Senior Citizen Status', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Senior Citizen (0=No, 1=Yes)')
        ax8.set_ylabel('Count')
        ax8.legend(title='Churn', labels=['No', 'Yes'])
        plt.setp(ax8.xaxis.get_majorticklabels(), rotation=0)
        
        # 9. Tech Support vs Churn
        ax9 = fig.add_subplot(gs[2, 2])
        tech_support_churn = pd.crosstab(self.df['TechSupport'], self.df['Churn'])
        tech_support_churn.plot(kind='bar', ax=ax9, color=['#2ecc71', '#e74c3c'])
        ax9.set_title('Churn by Tech Support', fontsize=12, fontweight='bold')
        ax9.set_xlabel('Tech Support')
        ax9.set_ylabel('Count')
        ax9.legend(title='Churn', labels=['No', 'Yes'])
        plt.setp(ax9.xaxis.get_majorticklabels(), rotation=45)
        
        plt.suptitle('Telco Customer Churn - Exploratory Data Analysis', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig('churn_eda_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'churn_eda_analysis.png'")
        plt.show()
    
    def create_correlation_heatmap(self):
        """Create correlation heatmap"""
        print("\nGenerating correlation heatmap...")
        
        # Create a copy and encode all categorical variables
        df_corr = self.df.copy()
        df_corr['Churn'] = (df_corr['Churn'] == 'Yes').astype(int)
        
        categorical_cols = df_corr.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'customerID':
                df_corr[col] = pd.factorize(df_corr[col])[0]
        
        # Select numeric columns only
        numeric_cols = df_corr.select_dtypes(include=[np.number]).columns
        
        # Create correlation matrix
        correlation_matrix = df_corr[numeric_cols].corr()
        
        # Plot heatmap
        plt.figure(figsize=(16, 12))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix - Telco Customer Churn', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("Correlation heatmap saved as 'correlation_heatmap.png'")
        plt.show()

def main():
    """Main execution"""
    eda = ChurnEDA('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Load data
    eda.load_data()
    
    # Perform analyses
    eda.basic_statistics()
    eda.numerical_analysis()
    eda.categorical_analysis()
    eda.churn_by_tenure()
    eda.churn_by_contract()
    eda.churn_by_internet_service()
    eda.correlation_analysis()
    
    # Create visualizations
    eda.create_visualizations()
    eda.create_correlation_heatmap()
    
    print("\n" + "="*60)
    print("EDA COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()