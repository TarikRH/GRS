# **GRS: Group Recommendation System using Spectral GCN, Contrastive Learning, and Distillation**

This project implements a Group Recommendation System (GRS) that leverages Spectral Graph Convolutional Networks (GCNs), Contrastive Learning, and Knowledge Distillation to provide item recommendations to clusters of users using the MovieLens dataset. The system is designed to handle both individual and group recommendations by clustering users and extracting group embeddings for improved recommendation accuracy.

======================================================================

## **Project Overview**

This project simulates a group-based recommendation system through the following components:

- Data Preprocessing: Transformation of MovieLens data into User-Item and User-User graphs.  
- User Clustering: Formation of user clusters based on interaction patterns and social connections.  
- Spectral GCN Module: Graph-based propagation of user-item and user-user interactions using spectral methods.  
- Contrastive Learning Module: Reinforces intra-cluster cohesion and inter-cluster separation using contrastive loss.  
- Distillation Module: Compresses high-dimensional embeddings into compact representations while maintaining structural information.  
- Evaluation: Assessment of model performance using metrics such as Recall@K and NDCG@K for both individual and group recommendations.

======================================================================
## **Key Features**

| Category             | Features                                        |
|----------------------|-------------------------------------------------|
| Data Processing      | Data cleaning, transformation, clustering      |
| Graph Construction   | User-Item and User-User graph generation       |
| AI-Powered GCN       | Spectral GCN with noise reduction              |
| Contrastive Learning | Intra-cluster and inter-cluster contrastive loss |
| Knowledge Distillation | Embedding compression and reduction         |
| Evaluation           | Recall@K, NDCG@K, Cluster cohesion             |


======================================================================
## **Tools**

| Category         | Technologies                       |
|------------------|------------------------------------|
| Data Processing  | Pandas, NumPy, Scikit-learn        |
| Graph Processing | NetworkX, PyTorch Geometric (PyG)  |
| ML/DL Models     | PyTorch, TensorFlow                |
| Clustering       | KMeans, Spectral Clustering        |
| Visualization    | Matplotlib, Seaborn                |
| Datasets         | MovieLens 1M                       |

======================================================================
## **Installation**

Prerequisites:

- Python 3.10+  
- PyTorch  
- PyTorch Geometric  
- MovieLens Dataset  

Setup:

1. Clone the repository:  

git clone https://github.com/TarikRH/GRS.git  
cd GRS  


======================================================================
## **Contributors**

- Mohamed Tarik Rhinaoui  

