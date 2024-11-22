# CyberGuard: IndiaAI Hackathon 🚀

## Project Overview
CyberGuard is an advanced Natural Language Processing (NLP) project designed to revolutionize cybercrime reporting systems. By accurately classifying fraud types based on victim descriptions, it addresses inefficiencies in existing processes. This project leverages cutting-edge models, unsupervised learning, manual annotation, and synthetic data generation for superior performance.

---

## Prerequisites

- **Python Version**: Ensure Python >= 3.9 is installed.
- **Setup Steps**:
  1. Activate your virtual environment:
     ```bash
     source env/bin/activate
     ```
  2. Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
  3. Run the inference script:
     ```bash
     python3 inference.py
     ```

---

## 🚩 Goal:
Build a series of machine learning models tailored for different levels of classification granularity to assist cybercrime categorization.

---

## Models Overview

### **Model 1: High-Level Main Category Classification**  
**Purpose**: Classify reports into one of the three `main_category` values:  
- Women/Child Related Crime  
- Financial Fraud Crimes  
- Other Cyber Crime  

---

### **Model 2: Women/Child-Related Crime Categories**
**Classes**:  
- Child Pornography/CSAM  
- Crime Against Women & Children  
- Online Cyber Trafficking  
- Rape/Gang Rape  
- Sexually Abusive Content  
- Sexually Obscene Material  

---

### **Model 3: Financial Fraud Crime Categories**
**Classes**:  
- Cryptocurrency Crime  
- Online Financial Fraud  
- Online Gambling/Betting  

---

### **Model 4: Other Cyber Crime Categories**
**Classes**:  
- Any Other Cyber Crime  
- Cyber Attack/Dependent Crimes  
- Cyber Terrorism  
- Hacking/System Damage  
- Online & Social Media Crime  
- Ransomware  
- Report Unlawful Content  

---

### **Model 5–10: Granular Sub-Categories**

#### Model 5: Sub-categories under **Hacking/Damage**
- Unauthorised Access/Data Breach  
- Email Hacking  
- Tampering with Computer Source Documents  
- Website Defacement/Hacking  
- System Damage  

#### Model 6: Sub-categories under **Online Financial Fraud**
- Fraud Call/Vishing  
- UPI-Related Frauds  
- Debit/Credit Card Fraud  
- e-Wallet Related Fraud  
- Demat Depository Fraud  
- Internet Banking Fraud  

#### Model 7: Sub-categories under **Other Cyber Crime**
- Identity Theft & Phishing  
- Zero-Day Exploits  
- Server/Network Attacks  
- Fake Mobile Apps  
- Ransomware  

#### Model 8: Sub-categories under **Crime Against Women & Children**
- Sexual Harassment  
- Cyber Blackmail/Threats  
- Computer-Generated CSAM  

#### Model 9: Sub-categories under **Cyber Attack/Dependent Crimes**
- SQL Injection  
- Malware Attack  
- Ransomware Attack  
- Data Breach/Theft  
- DDoS Attacks  

#### Model 10: Sub-categories under **Social Media-Related Crimes**
- Cyberbullying & Stalking  
- Fake Profiles  
- Online Job Fraud  
- Impersonation Email  

---

## Key Highlights ✨

1. **Data Quality Improvements**:  
   - Detected 30% mislabeling in the initial dataset.  
   - Applied clustering and manual annotation for corrections.

2. **Synthetic Data Generation**:  
   - Leveraged LLMs to enrich the dataset with realistic samples.

3. **BERT-Based Model**:  
   - Achieved precise classification across multiple levels.

---

## 🚀 Why CyberGuard Matters
By enabling accurate and granular classification of cybercrime reports, CyberGuard assists organizations in better analyzing trends, responding to incidents, and allocating resources effectively. This solution empowers victims, law enforcement, and analysts alike.

---

## Contribute
Want to improve or expand CyberGuard? Contributions are welcome!  
- **Fork this repository.**  
- **Create a new branch.**  
- **Submit a pull request.**

---

## License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.
