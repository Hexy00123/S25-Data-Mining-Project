import streamlit as st
import joblib
import pandas as pd
import implicit
from category_preprocessor import simplify_category_list, preprocess_text

# Set page configuration
st.set_page_config(
    page_title="ArXiv Journal Recommender",
    page_icon="📚",
    layout="wide"
)

# App title and description
st.title("📚 ArXiv Journal Recommender")
st.markdown("""
This app recommends scientific journals for your research paper based on its content.
Enter your paper details below to get personalized recommendations.
""")

# Load models and mappings
@st.cache_resource
def load_models_and_mappings():
    try:
        # Load classifier
        classifier = joblib.load('../models/arxiv_category_classifier_logreg.joblib')

        factors = 10
        regularization = 0.01
        iterations = 30
        calculate_training_loss = True
        use_gpu = implicit.gpu.HAS_CUDA

        print(f"Initializing ALS model with factors={factors}, regularization={regularization}, iterations={iterations}, use_gpu={use_gpu}")
        als_model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            calculate_training_loss=calculate_training_loss,
            use_gpu=use_gpu,
            random_state=42
        )
        
        # Load ALS model
        als_model = als_model.load('../models/als_model.npz')
        
        # Load matrix
        category_journal_matrix = joblib.load('../models/category_journal_matrix_PREDICTED.joblib')
        
        # Load mappings
        id2class_simplified = joblib.load('../models/id2class_simplified.joblib')
        class2id_simplified = {v: k for k, v in id2class_simplified.items()}
        
        journal_to_id = joblib.load('../models/journal_to_id.joblib')
        id_to_journal = {i: j for j, i in journal_to_id.items()}
        
        return {
            'classifier': classifier,
            'als_model': als_model,
            'category_journal_matrix': category_journal_matrix,
            'id2class_simplified': id2class_simplified,
            'class2id_simplified': class2id_simplified,
            'journal_to_id': journal_to_id,
            'id_to_journal': id_to_journal,
            'loaded': True
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {'loaded': False}

# Create input form
with st.form("paper_input_form"):
    st.header("Enter Paper Details")
    
    title = st.text_input("Paper Title")
    abstract = st.text_area("Paper Abstract", height=200)
    categories = st.text_input("arXiv Categories (optional, space separated, e.g., 'cs.AI math.AP')")
    
    submit_button = st.form_submit_button("Get Journal Recommendations")

if submit_button:
    if not title and not abstract:
        st.warning("Please enter at least a title or abstract for your paper.")
    else:
        with st.spinner("Processing your paper..."):
            models_data = load_models_and_mappings()
            
            if not models_data['loaded']:
                st.error("Failed to load models. Please check that all model files exist.")
                st.stop()
            
            # Process input text
            combined_text = f"{title} {abstract}"
            processed_text = preprocess_text(combined_text)
            
            # Get user-provided categories if any
            user_categories = simplify_category_list(categories) if categories else []
            
            classifier = models_data['classifier']
            predicted_categories = []
            
            try:
                category_predictions = classifier.predict([processed_text])
                
                if hasattr(category_predictions, 'tocoo'):
                    y_pred_coo = category_predictions.tocoo()
                    for _, cat_idx in zip(y_pred_coo.row, y_pred_coo.col):
                        if cat_idx in models_data['id2class_simplified']:
                            predicted_categories.append(models_data['id2class_simplified'][cat_idx])
            except Exception as e:
                st.warning(f"Category prediction error: {e}")
            
            # Use input categories if predictions failed
            if not predicted_categories and user_categories:
                predicted_categories = user_categories
            elif not predicted_categories and not user_categories:
                st.error("Could not determine categories for this paper.")
                st.stop()
            
            # Display categories
            st.subheader("Paper Categories")
            
            # Alternative display with Streamlit components
            col_container = st.container()
            cols = col_container.columns(min(len(predicted_categories), 4))
            
            for i, category in enumerate(predicted_categories):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    st.button(category, key=f"cat_{i}", disabled=True, 
                              help="This is a predicted category for your paper")
            
            als_model = models_data['als_model']
            class2id = models_data['class2id_simplified']
            id2journal = models_data['id_to_journal']
            category_journal_matrix = models_data['category_journal_matrix']
            
            recommended_journals = {}
            
            # For each category, get recommendations
            for category in predicted_categories:
                if category in class2id:
                    category_id = class2id[category]
                    
                    try:
                        ids, scores = als_model.recommend(
                            userid=category_id,
                            user_items=category_journal_matrix[category_id],
                            N=10,
                            filter_already_liked_items=False
                        )
                        
                        # Collect results
                        for journal_id, score in zip(ids, scores):
                            if journal_id in id2journal:
                                journal_name = id2journal[journal_id]
                                if journal_name in recommended_journals:
                                    recommended_journals[journal_name] += score
                                else:
                                    recommended_journals[journal_name] = score
                    except Exception as e:
                        st.warning(f"Error getting recommendations for category '{category}': {e}")
            
            st.header("Recommended Journals")
            
            if recommended_journals:
                sorted_journals = sorted(recommended_journals.items(), key=lambda x: x[1], reverse=True)
                
                result_df = pd.DataFrame(sorted_journals[:10], columns=["Journal", "Score"])
                result_df["Score"] = result_df["Score"].apply(lambda x: f"{x:.4f}")
                
                st.table(result_df)
            else:
                st.warning("No journal recommendations found for the given input.")