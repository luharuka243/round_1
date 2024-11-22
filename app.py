import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from loguru import logger
from ui_scripts.validate_csv import validate_input_csv
from ui_scripts.accuracy import accuracy_check


# Configure Loguru logger
logger.add("logs.txt", rotation="1 MB", retention="7 days", level="INFO")

logger.info("Streamlit app started")


# Load your model (replace with your model loading code)
def load_model():
    # Placeholder for model loading
    model = None  # Replace with actual model
    return model

# Function to make predictions
def predict(data, model):
    # Replace with your model's prediction logic
    predictions = [{"main_category":"O1","category":"02","sub_category":"03"}] * len(data)  # Dummy output
    return predictions

# Load the model
model = load_model()

# Sidebar for page navigation
st.sidebar.title("Jump to Section")
page = st.sidebar.radio("Go to", ["Model Predictions", "Documentation","About CloudSEK"])

if page == "Model Predictions":
    st.markdown(
        """
        <style>
        body {
            background-color: #121212; /* Dark background for sleek look */
            color: #EDEDED; /* Light text color for contrast */
            font-family: 'Roboto', sans-serif; /* Modern font */
        }
        h1, h2, h3 {
            color: #80C7E2; /* Soft Cyan for titles and headers */
        }
        .header-text {
            font-size: 18px;
            color: #A4B3B9; /* Lighter gray text for paragraphs */
        }
        .section-title {
            color: #4CAF50; /* Green for section titles for a refreshing pop */
            font-weight: bold;
        }
        .subheader-text {
            color: #B0BEC5;
            font-size: 16px;
        }
        .input-label {
            color: #90CAF9; /* Lighter cyan for the input labels */
            font-size: 14px;
        }
        .prediction-box {
            background-color: #2C2F36; /* Slightly lighter dark background for predictions */
            border: 1px solid #424242;
            padding: 15px;
            border-radius: 8px;
            color: #80C7E2;
            font-size: 16px;
        }
        .button {
            background-color: #00796B; /* Teal button for a unique look */
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            cursor: pointer;
        }
        .button:hover {
            background-color: #004D40;
        }
        .file-upload-box {
            background-color: #333333; /* Dark box for file upload */
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #424242;
        }
        .download-btn {
            background-color: #4CAF50; /* Green button for download */
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            cursor: pointer;
        }
        .download-btn:hover {
            background-color: #388E3C;
        }
        .title-text {
            color: #FFFFFF; /* White color for the title */
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    st.markdown('<h1 class="title-text">IndiaAI CyberGuard AI Hackathon</h1>', unsafe_allow_html=True)

    # Real-time model check section
    st.header("Real-Time Model Check")
    input_text = st.text_input("Enter your input data:", label_visibility="collapsed")
    if st.button("Predict", key="real-time-predict", help="Click to get real-time predictions"):
        if input_text:
            # Add model prediction logic for real-time inputs
            prediction = predict([input_text], model)
            st.markdown(f"<div class='prediction-box'>Prediction: {prediction}</div>", unsafe_allow_html=True)
        else:
            st.warning("Please enter some input data.")

    # CSV file input for batch processing
    st.header("Batch Prediction from CSV")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="batch-upload")
    logger.info(f"{uploaded_file}")
    if uploaded_file:
        # Read the uploaded CSV
        data = pd.read_csv(uploaded_file)

        # Validate CSV input
        if not validate_input_csv(data):
            st.error("Invalid input CSV. Please check the format and content.")
        
        # Display the uploaded data
        st.write("Uploaded Data:")
        st.write(data.head(3))

        # Dropdown to show the required columns for CSV
        required_columns = ['id','input_text','ground_truth [Optional for accuracy]']  # List required columns
        st.subheader("Required Columns in CSV:")
        st.markdown(f"<span class='subheader-text'>The uploaded CSV should contain the following columns:</span>", unsafe_allow_html=True)
        st.write(required_columns)

        # # Dropdown for users to select columns from the uploaded CSV
        # selected_column = st.selectbox("Select the input column from the uploaded CSV", options=data.columns.tolist())
        # st.write(f"Selected Column: {selected_column}")

        # Dropdown to choose between models
        selected_model = st.selectbox("Choose the model", ["Model 1", "Model 2"], key="model-selection")
        

        if st.button("Process CSV", key="process-csv", help="Click to process the uploaded CSV for predictions"):
            # Check if the selected column exists in the uploaded CSV
            predictions = predict(data['input_text'], model)
            data['Prediction'] = predictions
            st.write("Prediction Results CSV looks like this")
            st.write(data.head(2))

            if 'ground_truth' in data.columns:
                logger.info('Ground truth provided')
                accuracy=accuracy_check(predictions,list(dataframe['ground_truth']))
                st.write(f"Model Accuracy on given CSV is : {accuracy}%")
            # Option to download the output CSV with predictions
            csv = data.to_csv(index=False)
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
                key="download-btn",
                help="Click to download the CSV file with predictions"
            )

elif page == "Documentation":
    st.markdown(
        """
        <style>
        body {
            background-color: #121212; /* Dark background for sleek look */
            color: #EDEDED; /* Light text color for contrast */
            font-family: 'Roboto', sans-serif; /* Modern font */
        }
        h1, h2, h3 {
            color: #80C7E2; /* Soft Cyan for titles and headers */
        }
        .header-text {
            font-size: 18px;
            color: #A4B3B9; /* Lighter gray text for paragraphs */
        }
        .section-title {
            color: #4CAF50; /* Green for section titles for a refreshing pop */
            font-weight: bold;
        }
        .subheader-text {
            color: #B0BEC5;
            font-size: 16px;
        }
        .input-label {
            color: #90CAF9; /* Lighter cyan for the input labels */
            font-size: 14px;
        }
        .prediction-box {
            background-color: #2C2F36; /* Slightly lighter dark background for predictions */
            border: 1px solid #424242;
            padding: 15px;
            border-radius: 8px;
            color: #80C7E2;
            font-size: 16px;
        }
        .button {
            background-color: #00796B; /* Teal button for a unique look */
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            cursor: pointer;
        }
        .button:hover {
            background-color: #004D40;
        }
        .file-upload-box {
            background-color: #333333; /* Dark box for file upload */
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #424242;
        }
        .download-btn {
            background-color: #4CAF50; /* Green button for download */
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            cursor: pointer;
        }
        .download-btn:hover {
            background-color: #388E3C;
        }
        .title-text {
            color: #FFFFFF; /* White color for the title */
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown('<h1 class="title-text">Model Documentation and Analysis</h1>', unsafe_allow_html=True)

    # About the model section
    st.header("About the Model")
    st.markdown("""
    <style>
        .header-text {
            font-size: 18px;
            font-weight: bold;
            color: white;
            margin-bottom: 15px;
        }
        .section-title {
            color: #4CAF50; /* Highlighted color for better visibility */
        }
        ul {
            color: white;
            margin-left: 20px;
        }
        li {
            margin-bottom: 10px;
        }
    </style>
    <p class="header-text">This model is designed to solve a specific problem. It uses advanced machine learning techniques to process data and generate accurate predictions. Below are the key aspects of the model:</p>
    <ul>
        <li><span class="section-title">Input Format:</span> 
            <ul>
                <li>Data: <code>train.csv</code> with over 80,000 data points.</li>
                <li><b>Main Categories:</b> High-level crime categories:
                    <ul>
                        <li>"Women/Child Related Crime"</li>
                        <li>"Financial Fraud Crimes"</li>
                        <li>"Other Cyber Crime"</li>
                    </ul>
                </li>
                <li><b>Category:</b> Specific crime types within each main category.</li>
                <li><b>Subcategories:</b> Further granularity based on categories.</li>
            </ul>
            To establish relationships, we use:
            <ul>
                <li><b>Main Category to Category Mapping:</b> Maps each `main_category` to corresponding `category` values.</li>
                <li><b>Category to Subcategory Mapping:</b> Maps each `category` to its `sub_category` values.</li>
            </ul>
        </li>
        <li><span class="section-title">Models:</span> A series of machine learning models tailored for different levels of classification granularity:
            <ul>
                <li><b>Model 1:</b> Classifies data into one of three `main_category` values:
                    <ul>
                        <li>"Women/Child Related Crime"</li>
                        <li>"Financial Fraud Crimes"</li>
                        <li>"Other Cyber Crime"</li>
                    </ul>
                </li>
                <li><b>Model 2:</b> Classifies data into categories under "Women/Child Related Crime," such as:
                    <ul>
                        <li>"Child Pornography/CSAM"</li>
                        <li>"Crime Against Women & Children"</li>
                        <li>"Online Cyber Trafficking"</li>
                        <li>"Sexually Obscene Material"</li>
                        <li>"Rape/Gang Rape"</li>
                        <li>"Sexually Abusive Content"</li>
                    </ul>
                </li>
                <li><b>Model 3:</b> Classifies categories under "Financial Fraud Crimes," such as:
                    <ul>
                        <li>"Cryptocurrency Crime"</li>
                        <li>"Online Financial Fraud"</li>
                        <li>"Online Gambling/Betting"</li>
                    </ul>
                </li>
                <li><b>Model 4:</b> Classifies categories under "Other Cyber Crime," including:
                    <ul>
                        <li>"Any Other Cyber Crime"</li>
                        <li>"Cyber Attack/Dependent Crimes"</li>
                        <li>"Cyber Terrorism"</li>
                        <li>"Hacking/Damage to Computer Systems"</li>
                        <li>"Online and Social Media Related Crime"</li>
                        <li>"Ransomware"</li>
                        <li>"Report Unlawful Content"</li>
                    </ul>
                </li>
                <li><b>Model 5:</b> Classifies subcategories under "Hacking/Damage to Computer Systems," such as:
                    <ul>
                        <li>"Unauthorized Access/Data Breach"</li>
                        <li>"Email Hacking"</li>
                        <li>"Tampering with Computer Source Documents"</li>
                        <li>"Website Defacement/Hacking"</li>
                        <li>"Damage to Computer Systems"</li>
                    </ul>
                </li>
                <li><b>Model 6:</b> Classifies subcategories under "Online Financial Fraud," such as:
                    <ul>
                        <li>"Demat/Depository Fraud"</li>
                        <li>"Fraud Call/Vishing"</li>
                        <li>"Internet Banking Related Fraud"</li>
                        <li>"Business Email Compromise/Email Takeover"</li>
                        <li>"UPI Related Frauds"</li>
                        <li>"Debit/Credit Card Fraud/SIM Swap Fraud"</li>
                        <li>"E-Wallet Related Fraud"</li>
                    </ul>
                </li>
                <li><b>Model 7:</b> Classifies subcategories under "Any Other Cyber Crime," such as:
                    <ul>
                        <li>"Identity Theft, Spoofing, and Phishing Attacks"</li>
                        <li>"Zero-Day Exploits"</li>
                        <li>"Attacks on Servers and Networks"</li>
                        <li>"Web Application Vulnerabilities"</li>
                        <li>"Attacks on Critical Infrastructure"</li>
                        <li>"Disinformation or Misinformation Campaigns"</li>
                    </ul>
                </li>
                <li><b>Model 8:</b> Classifies subcategories under "Crime Against Women & Children," such as:
                    <ul>
                        <li>"Sexual Harassment"</li>
                        <li>"Computer Generated CSAM/CSEM"</li>
                        <li>"Cyber Blackmailing & Threatening"</li>
                    </ul>
                </li>
                <li><b>Model 9:</b> Classifies subcategories under "Cyber Attack/Dependent Crimes," such as:
                    <ul>
                        <li>"Hacking/Defacement"</li>
                        <li>"SQL Injection"</li>
                        <li>"Malware Attack"</li>
                        <li>"Denial of Service (DoS)/DDoS Attacks"</li>
                        <li>"Ransomware Attack"</li>
                        <li>"Data Breach/Theft"</li>
                    </ul>
                </li>
                <li><b>Model 10:</b> Classifies subcategories under "Online and Social Media Related Crime," such as:
                    <ul>
                        <li>"Cyber Bullying, Stalking, Sexting"</li>
                        <li>"Impersonating Email"</li>
                        <li>"Profile Hacking/Identity Theft"</li>
                        <li>"Online Job Fraud"</li>
                        <li>"Provocative Speech for Unlawful Acts"</li>
                        <li>"Fake/Impersonating Profile"</li>
                        <li>"Online Matrimonial Fraud"</li>
                    </ul>
                </li>
            </ul>
        </li>
        <li><span class="section-title">Architecture:</span> The models use advanced architectures such as CNNs, RNNs, or Transformers, depending on the classification task.</li>
    </ul>
    """, unsafe_allow_html=True)

    st.subheader("Categories Classifiers")
    with open("ui_images/ai_hack_flowsvg_02.svg", "r") as svg_file:
        svg_content = svg_file.read()
    st.image(svg_content, use_column_width=True)
    # Load and display the SVG
    st.subheader("Re-imagining the clusters")
    with open("ui_images/ai_hack_flowsvg_04.svg", "r") as svg_file:
        svg_content = svg_file.read()
    st.image(svg_content, use_column_width=True)

    # Accuracy over epochs graph
    st.subheader("Accuracy Over Epochs")
    epochs = np.arange(1, 11)  # Example epoch numbers
    accuracy = np.random.uniform(0.7, 1.0, len(epochs))  # Simulated accuracy values

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracy, marker='o', color='blue', linewidth=2, markersize=8)
    plt.title("Model Accuracy Over Epochs", fontsize=18, color='#000000', weight='bold')
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(plt)

    # Loss over epochs graph
    st.subheader("Loss Over Epochs")
    loss = np.random.uniform(0.1, 0.5, len(epochs))  # Simulated loss values

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, marker='o', color='red', linewidth=2, markersize=8)
    plt.title("Model Loss Over Epochs", fontsize=18, color='#000000', weight='bold')
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(plt)

    # New Section: Deployment and Scaling
    st.header("Deployment and Scaling")

    st.markdown("""
        <style>
            .content-text {
                font-size: 16px;
                line-height: 1.6;
                color: #f0f0f0;
            }
            .section-title {
                font-weight: bold;
                font-size: 18px;
                color: #80C7E2;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <p class="content-text">
        For deploying the BERT-based text classifier, we will use a Dockerized container orchestrated with Kubernetes, ensuring efficient management and scaling of the application. The deployment will run on multiple pods for high availability and reliability.
        For scaling, we primarily control the deployment using resource requests, limits, and replicas, allowing us to balance performance and cost effectively.
        In Kubernetes, managing container resources effectively ensures optimal performance and prevents resource overutilization. Below are the key configurations:
        </p>
        <p class="content-text">
            In this case:
        </p>
        <ul>
            <li><span class="section-title">CPU:</span> Minimum 500m (50% of a core), Maximum 1 (100% of a core).</li>
            <li><span class="section-title">Memory:</span> Minimum 2Gi, Maximum 4Gi.</li>
        </ul>
        <p class="content-text">
            Proper configuration helps in maintaining service availability and scaling efficiently under load.
        </p>
    """, unsafe_allow_html=True)

    # Display Flowchart
    flowchart_path = "ui_images/deployment_fw_01.png"  # Update with the actual path to your flowchart
    st.image(flowchart_path, caption="Flowchart: Resource Allocation in Kubernetes", use_column_width=True)

    # Dockerfile and Kubernetes Deployment YAML Details
    st.markdown("""
        <p class="content-text">
            <span class="section-title">Dockerfile:</span> 
            The Dockerfile configures the environment for model serving:
        </p>
        <ul>
            <li>Uses Python 3.9 slim image</li>
            <li>Installs necessary dependencies</li>
            <li>Exposes port 8080</li>
            <li>Sets up environment for model serving</li>
        </ul>

        <p class="content-text">
            <span class="section-title">Kubernetes Deployment YAML:</span> 
            The Kubernetes configuration ensures scalability and availability:
        </p>
        <ul>
            <li>3 replica deployment</li>
            <li>Resource requests and limits</li>
            <li>Persistent volume claim for model storage</li>
            <li>LoadBalancer service type</li>
        </ul>
    """, unsafe_allow_html=True)

    # Additional Insights section
    st.header("Future Scope ")
    # st.subheader('Heterogeneous graph neural network (HeteroGNN)')
    st.markdown('<h1 class="title-text">Heterogeneous graph neural network (HeteroGNN)</h1>', unsafe_allow_html=True)
    # Detailed insights about the model
    st.markdown(
        """
        <style>
            .content-text {
                font-size: 16px;
                line-height: 1.6;
                color: #f0f0f0;
            }
            .section-title {
                font-weight: bold;
                font-size: 18px;
                color: #80C7E2;
            }
            .list-item {
                margin-bottom: 10px;
            }
        </style>
        """, unsafe_allow_html=True
    )
    st.markdown(
        """
        <p class="content-text">Explore detailed insights about the potential advancements with HeteroGNN:</p>
        <ol>
            <li class="list-item">
                <span class="section-title">Key Model Components and Architecture:</span>
                <p class="content-text">
                    A heterogeneous graph is a graph with multiple types of nodes and edges. In our text classification context:
                </p>
                <ul>
                    <li><strong>Node Types:</strong></li>
                    <ul>
                        <li>1. <strong>Document Nodes:</strong> Represent individual text documents</li>
                        <li>2. <strong>Label Nodes:</strong> Represent unique category labels</li>
                    </ul>
                    <li><strong>Edge Types:</strong></li>
                    <ul>
                        <li>
                            1. <strong>Document-Document Edges:</strong><br>
                            - Created based on document text similarity<br>
                            - Connects semantically similar documents<br>
                            - Helps capture contextual relationships
                        </li>
                        <li>
                            2. <strong>Document-Label Edges:</strong><br>
                            - Connects documents to their ground truth labels<br>
                            - Guides the learning process during training
                        </li>
                    </ul>
                </ul>
            </li>
            <li class="list-item">
                <span class="section-title">Examples of Inputs and Outputs:</span>
                <p class="content-text">
                    <strong>A. Text Embedding:</strong><br>
                    - Uses SentenceTransformer ('all-MiniLM-L6-v2')<br>
                    - Converts raw text into dense vector representations<br>
                    - Captures semantic meaning in a fixed-length vector<br>
                    - Enables meaningful similarity comparisons
                </p>
                <p class="content-text">
                    <strong>B. Graph Construction:</strong>
                </p>
                <ul>
                    <li>
                        <strong>Document Node Features:</strong><br>
                        - Embedding vectors from sentence transformer<br>
                        - Captures semantic content of each document
                    </li>
                    <li>
                        <strong>Label Node Features:</strong><br>
                        - One-hot encoded vectors<br>
                        - Represents unique category information
                    </li>
                    <li>
                        <strong>Edge Creation:</strong>
                        <ul>
                            <li>
                                <strong>Document-Document Edges:</strong><br>
                                - Uses cosine similarity<br>
                                - Connects top-k most similar documents<br>
                                - Helps propagate information between related texts
                            </li>
                            <li>
                                <strong>Document-Label Edges:</strong><br>
                                - Connects documents to their ground truth labels<br>
                                - Provides supervised learning signal
                            </li>
                        </ul>
                    </li>
                </ul>
            </li>
            <li class="list-item">
                <span class="section-title">Advanced Visualizations:</span>
                <p class="content-text">
                    Below are potential visualizations illustrating how the model processes graphs:<br>
                    - Node embedding distributions<br>
                    - Attention weights for relation-specific edges<br>
                </p>
            </li>
        </ol>
        """, unsafe_allow_html=True
    )
    # Add images and text
    image1 = Image.open("ui_images/gnn_fw_2.png")
    st.image(image1, caption="Model Architecture", use_column_width=True)
    image2 = Image.open("ui_images/gnn_fw_1.png")
    st.image(image2, caption="Model inference", use_column_width=True)

    st.markdown("""
        <p class="content-text">These visualizations help provide a clearer understanding of the model's design and performance.</p>
        <p class="content-text">However, heterogeneous graph neural networks (HeteroGNNs) might not be the best fit for certain scenarios due to the following limitations:</p>
        <ul>
            <li><span class="section-title">Complexity in Graph Construction:</span> Building heterogeneous graphs requires defining multiple node and edge types, which can lead to significant preprocessing overhead.</li>
            <li><span class="section-title">Scalability Challenges:</span> As the graph size increases, processing and training become computationally intensive, potentially exceeding resource limitations.</li>
            <li><span class="section-title">Sparse Data Issues:</span> If the dataset lacks sufficient diversity or connectivity between different node types, the graph may fail to generalize well.</li>
            <li><span class="section-title">Interpretability Concerns:</span> Understanding how different edge types and node interactions contribute to the final prediction can be challenging, making it harder to debug or explain the model.</li>
        </ul>
        <p class="content-text">Given these challenges, alternative approaches such as homogeneous graph networks or traditional machine learning techniques might be more practical and effective for our current requirements.</p>
        <p class="content-text">For more details, refer to the documentation or reach out to our team.</p>
    """, unsafe_allow_html=True)

elif page == "About CloudSEK":
    st.markdown(
        """
        <style>
        body {
            background-color: #1a1a1a; /* Dark background */
            color: #f0f0f0; /* Light text color */
            font-family: 'Arial', sans-serif; /* Font */
        }
        h1 {
            color: #FFFFFF; /* White color for the main title */
        }
        h2 {
            color: #80C7E2; /* Soft cyan color for subheadings */
        }
        p {
            line-height: 1.6; /* Increase line height for readability */
        }
        .title-text {
            color: #FFFFFF; /* White color for the title */
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    # Apply the custom class to the main title
    st.markdown('<h1 class="title-text">CloudSEK</h1>', unsafe_allow_html=True)
    
    # # About the company
    # st.header("About CloudSEK")
    st.markdown("""
    At CloudSEK, we combine the power of Cyber Intelligence, Brand Monitoring, Attack Surface Monitoring, Infrastructure Monitoring, and Supply Chain Intelligence to provide a comprehensive view of digital risks. Our offerings include:
    - **Cyber Intelligence**: Utilizing advanced machine learning techniques to analyze vast amounts of data to uncover patterns and trends in cyber threats.
    - **Comprehensive Assets Tracker**: Monitor all digital assets across various platforms to ensure thorough protection against external threats.
    - **Surface, Deep, and Dark Web Monitoring**: Continuously scan the internet, including the surface, deep, and dark web, for potential threats and mentions of your organization.
    - **Integrated Threat Intelligence**: Combine threat intelligence from multiple sources for a comprehensive understanding of the external threat landscape.
    """, unsafe_allow_html=True)

    # About Team Section
    st.header("About Our Team")
    st.markdown("""
    - **Lasya Ippagunta**: [LinkedIn Profile](https://www.linkedin.com)
    - **Apurv Singh**: [LinkedIn Profile](https://www.linkedin.com/in/apurvsj/)
    - **Shubham Luharuka**: [LinkedIn Profile](https://www.linkedin.com/in/shubhamluharuka/)
    - **Sravanthi P**: [LinkedIn Profile](https://www.linkedin.com/in/p-l-sravanthi-b23360217/)
    - **Puneet Hedge**: [LinkedIn Profile](https://www.linkedin.com/in/puneetthegde/)
    - **Faizah Feroz**: [LinkedIn Profile](https://www.linkedin.com)
    - **HS Manoj**: [LinkedIn Profile](https://www.linkedin.com/in/manojkumarhs/)
    """, unsafe_allow_html=True)