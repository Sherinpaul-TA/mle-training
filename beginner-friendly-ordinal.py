# BEGINNER-FRIENDLY FRANK & HALL ORDINAL CLASSIFICATION
# ====================================================
# This code is SUPER SIMPLE but keeps ALL the important concepts!

# Step 1: Import all the tools we need (like importing tools from a toolbox)
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

# =============================================================================
# STEP 1: THE MAIN CLASS - THINK OF THIS AS A SMART MACHINE
# =============================================================================

class SimpleOrdinalClassifier(BaseEstimator, ClassifierMixin):
    """
    🤖 SIMPLE ORDINAL CLASSIFIER - Easy to Understand!
    
    WHAT THIS DOES (Core Concept):
    Instead of treating classes 0,1,2,3,4,5,6 as completely different categories,
    we ask 6 simple YES/NO questions:
    
    Question 1: "Is this class bigger than 0?" 
    Question 2: "Is this class bigger than 1?"
    Question 3: "Is this class bigger than 2?"  
    Question 4: "Is this class bigger than 3?"
    Question 5: "Is this class bigger than 4?"
    Question 6: "Is this class bigger than 5?"
    
    WHY THIS IS SMART:
    - If something is class 3, it will answer YES to questions 1,2,3 and NO to 4,5,6
    - This keeps the natural order: 0 < 1 < 2 < 3 < 4 < 5 < 6
    - Much smarter than treating them as random categories!
    """
    
    def __init__(self, n_trees=50, tree_depth=3, learning_speed=0.1):
        """
        🏗️ SETUP: Tell the machine how to learn
        
        Parameters (like settings on your phone):
        - n_trees: How many decision trees to use (more = smarter but slower)
        - tree_depth: How deep each tree can grow (deeper = more complex)
        - learning_speed: How fast to learn (slower = more careful)
        """
        self.n_trees = n_trees
        self.tree_depth = tree_depth  
        self.learning_speed = learning_speed
        
        # These will store our 6 YES/NO question machines
        self.question_machines = []
        self.num_classes = None
    
    def fit(self, X, y):
        """
        🎓 TRAINING: Teach the machine using your data
        
        X = your input data (like height, weight, age, etc.)
        y = the correct answers (classes 0,1,2,3,4,5,6)
        """
        print("🤖 Training the Smart Ordinal Classifier...")
        
        # Remember how many classes we have
        self.num_classes = len(np.unique(y))
        print(f"   📚 Learning {self.num_classes} different classes")
        
        # Clear any old machines
        self.question_machines = []
        
        # Create 6 YES/NO question machines (for 7 classes)
        for question_num in range(self.num_classes - 1):  # 0,1,2,3,4,5
            
            print(f"   🔍 Training question {question_num + 1}: 'Is class > {question_num}?'")
            
            # Convert the multi-class problem into YES/NO
            # Example: if question_num=2, then classes 0,1,2 → NO, classes 3,4,5,6 → YES
            yes_no_answers = (y > question_num).astype(int)
            
            # Count how many YES vs NO answers
            yes_count = np.sum(yes_no_answers == 1)
            no_count = np.sum(yes_no_answers == 0) 
            print(f"      📊 YES answers: {yes_count}, NO answers: {no_count}")
            
            # Create a smart XGBoost machine for this YES/NO question
            question_machine = XGBClassifier(
                objective='binary:logistic',  # This means YES/NO question
                n_estimators=self.n_trees,
                max_depth=self.tree_depth,
                learning_rate=self.learning_speed,
                reg_alpha=1.0,      # Regularization (prevents overlearning)
                reg_lambda=2.0,     # More regularization 
                random_state=42     # Make results repeatable
            )
            
            # Train this machine on the YES/NO answers
            question_machine.fit(X, yes_no_answers)
            
            # Save this trained machine
            self.question_machines.append(question_machine)
        
        print(f"   ✅ Training complete! Created {len(self.question_machines)} question machines")
        return self
    
    def predict_proba(self, X):
        """
        🔮 PROBABILITY PREDICTION: How confident is the machine?
        
        This is the CLEVER PART - converting YES/NO answers back to class probabilities!
        """
        num_samples = X.shape[0]  # How many things to predict
        all_probabilities = np.zeros((num_samples, self.num_classes))
        
        # Ask each question machine for its YES probability
        yes_probabilities = []
        for machine in self.question_machines:
            # Get probability of YES for each question
            prob_yes = machine.predict_proba(X)[:, 1]  # [:,1] means "YES" probability
            yes_probabilities.append(prob_yes)
        
        # MAGIC CONVERSION: Turn YES/NO probabilities into class probabilities
        for i in range(num_samples):  # For each sample
            
            # Probability of class 0 = 1 - P(class > 0)  
            all_probabilities[i, 0] = 1 - yes_probabilities[0][i]
            
            # Probability of middle classes = P(class > j-1) - P(class > j)
            for class_num in range(1, self.num_classes - 1):
                all_probabilities[i, class_num] = (yes_probabilities[class_num-1][i] - 
                                                 yes_probabilities[class_num][i])
            
            # Probability of highest class = P(class > 5)
            all_probabilities[i, self.num_classes - 1] = yes_probabilities[-1][i]
            
            # IMPORTANT: Make sure probabilities are valid (between 0 and 1, sum to 1)
            all_probabilities[i] = np.maximum(all_probabilities[i], 0)  # No negative probs
            prob_total = np.sum(all_probabilities[i])
            if prob_total > 0:
                all_probabilities[i] = all_probabilities[i] / prob_total  # Make sum = 1
            else:
                # If something goes wrong, give equal probability to all classes
                all_probabilities[i] = np.ones(self.num_classes) / self.num_classes
        
        return all_probabilities
    
    def predict(self, X):
        """
        🎯 FINAL PREDICTION: Which class is most likely?
        """
        probabilities = self.predict_proba(X)
        # Return the class with highest probability
        return np.argmax(probabilities, axis=1)

print("✅ Simple Ordinal Classifier class loaded!")

# =============================================================================
# STEP 2: SIMPLE TRAINING FUNCTION
# =============================================================================

def train_simple_ordinal_model(X_train, X_test, y_train, y_test, quick_mode=True):
    """
    🚀 ONE-BUTTON TRAINING: Does everything automatically!
    
    What this function does:
    1. Scales your data (makes numbers comparable)
    2. Balances classes (gives fair weight to all classes)  
    3. Tries different settings to find the best ones
    4. Tests the final model
    5. Shows you the results
    
    Parameters:
    X_train, X_test = your data (features like height, age, income, etc.)
    y_train, y_test = correct answers (must be 0,1,2,3,4,5,6)
    quick_mode = True for fast testing, False for thorough search
    """
    
    print("🚀 STARTING SIMPLE ORDINAL CLASSIFICATION TRAINING")
    print("=" * 60)
    
    # STEP 1: Check your data
    print("📋 Checking your data...")
    print(f"   Training samples: {X_train.shape[0]:,}")
    print(f"   Test samples: {X_test.shape[0]:,}")
    print(f"   Features (columns): {X_train.shape[1]:,}")
    
    # Make sure classes are correct format
    unique_classes = np.unique(y_train)
    print(f"   Classes found: {unique_classes}")
    
    if not np.array_equal(unique_classes, np.arange(len(unique_classes))):
        print("   ❌ ERROR: Your classes must be 0,1,2,3,4,5,6!")
        print("   💡 Fix with: y_train = y_train - min(y_train)")
        return None
    
    print("   ✅ Data looks good!")
    print()
    
    # STEP 2: Scale the data (make all numbers similar size)
    print("⚖️ Scaling your data...")
    scaler = StandardScaler()  # This makes all features have similar ranges
    X_train_scaled = scaler.fit_transform(X_train)  # Learn scaling from training
    X_test_scaled = scaler.transform(X_test)        # Apply same scaling to test
    print("   ✅ Data scaled (all features now have similar ranges)")
    print()
    
    # STEP 3: Balance the classes (give fair treatment to all)
    print("⚖️ Calculating class balance weights...")
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    sample_weights = np.array([class_weights[label] for label in y_train])
    
    print("   Class weights (higher = more attention needed):")
    for i, weight in enumerate(class_weights):
        count = np.sum(y_train == i)
        print(f"     Class {i}: weight={weight:.2f} (has {count} samples)")
    print()
    
    # STEP 4: Try different settings to find the best one
    print("🔍 Searching for best settings...")
    
    if quick_mode:
        # Small search for quick testing
        settings_to_try = {
            'n_trees': [30, 50],           # Try 30 or 50 trees
            'tree_depth': [3, 4],          # Try depth 3 or 4
            'learning_speed': [0.1, 0.15]  # Try these learning speeds
        }
        print("   Using QUICK search (faster, good for testing)")
    else:
        # More thorough search
        settings_to_try = {
            'n_trees': [30, 50, 80],
            'tree_depth': [3, 4, 5],
            'learning_speed': [0.05, 0.1, 0.15]
        }
        print("   Using THOROUGH search (slower, better results)")
    
    total_combinations = 1
    for key, values in settings_to_try.items():
        total_combinations *= len(values)
    print(f"   Will try {total_combinations} different combinations")
    print()
    
    # STEP 5: Create the classifier and search for best settings
    print("🤖 Training models with different settings...")
    
    # Create our smart classifier
    classifier = SimpleOrdinalClassifier()
    
    # GridSearchCV = automatic testing of different settings
    grid_search = GridSearchCV(
        classifier,                    # Our classifier
        settings_to_try,              # Settings to try
        cv=3,                         # Test each setting 3 times (cross-validation)
        scoring='accuracy',           # Judge by accuracy
        n_jobs=-1,                    # Use all computer cores (faster)
        verbose=1                     # Show some progress
    )
    
    # Start training! (This is where the magic happens)
    grid_search.fit(X_train_scaled, y_train)
    
    print("   ✅ Training completed!")
    print()
    
    # STEP 6: Get the best model
    print("🏆 Finding the best model...")
    best_model = grid_search.best_estimator_
    best_settings = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print("   🎯 Best settings found:")
    for setting, value in best_settings.items():
        print(f"     {setting}: {value}")
    print(f"   📊 Best training score: {best_score:.3f}")
    print()
    
    # STEP 7: Test on unseen data
    print("📊 Testing on unseen data...")
    
    # Make predictions
    test_predictions = best_model.predict(X_test_scaled)
    test_probabilities = best_model.predict_proba(X_test_scaled)
    
    # Calculate how good it is
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    
    # Check if it's overfitting (memorizing vs learning)
    train_predictions = best_model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, train_predictions)
    overfitting = train_accuracy - test_accuracy
    
    print("   📈 FINAL RESULTS:")
    print(f"     Test Accuracy: {test_accuracy:.3f} (higher is better)")
    print(f"     Test MAE: {test_mae:.3f} (lower is better)")
    print(f"     Training Accuracy: {train_accuracy:.3f}")
    print(f"     Overfitting Gap: {overfitting:.3f}")
    
    # Explain overfitting
    if overfitting < 0.05:
        print("     ✅ Excellent! Model generalizes well")
    elif overfitting < 0.10:
        print("     ✅ Good! Small overfitting")
    else:
        print("     ⚠️ Some overfitting detected")
    print()
    
    # STEP 8: Show prediction examples
    print("🔮 Example predictions (first 10):")
    print("   Actual → Predicted (Confidence)")
    for i in range(min(10, len(y_test))):
        actual = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
        predicted = test_predictions[i]
        confidence = np.max(test_probabilities[i])
        print(f"     {actual} → {predicted} ({confidence:.2f})")
    print()
    
    # Return everything important
    results = {
        'model': best_model,
        'scaler': scaler,
        'test_accuracy': test_accuracy,
        'test_mae': test_mae,
        'overfitting': overfitting,
        'best_settings': best_settings,
        'predictions': test_predictions,
        'probabilities': test_probabilities
    }
    
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return results

print("✅ Simple training function loaded!")

# =============================================================================
# STEP 3: SIMPLE PREDICTION FUNCTION
# =============================================================================

def predict_new_data(model, scaler, X_new):
    """
    🔮 PREDICT NEW DATA: Use your trained model on new samples
    
    Parameters:
    model = your trained classifier
    scaler = the scaler used during training
    X_new = new data to predict (same format as training data)
    """
    
    print("🔮 MAKING PREDICTIONS ON NEW DATA")
    print("=" * 40)
    
    # Scale new data the same way as training data
    print("⚖️ Scaling new data...")
    X_new_scaled = scaler.transform(X_new)
    print(f"   ✅ {X_new.shape[0]} samples scaled")
    
    # Make predictions
    print("🤖 Generating predictions...")
    predictions = model.predict(X_new_scaled)
    probabilities = model.predict_proba(X_new_scaled)
    
    # Calculate confidence
    confidence_scores = np.max(probabilities, axis=1)
    average_confidence = np.mean(confidence_scores)
    
    print(f"   ✅ Predictions completed!")
    print(f"   📊 Average confidence: {average_confidence:.3f}")
    
    # Show summary
    print("\n📊 Prediction Summary:")
    for class_num in range(model.num_classes):
        count = np.sum(predictions == class_num)
        percentage = count / len(predictions) * 100
        print(f"   Class {class_num}: {count} samples ({percentage:.1f}%)")
    
    # Show first few predictions
    print(f"\n🔍 First {min(5, len(predictions))} predictions:")
    for i in range(min(5, len(predictions))):
        pred_class = predictions[i]
        confidence = confidence_scores[i]
        print(f"   Sample {i+1}: Class {pred_class} (confidence: {confidence:.3f})")
    
    return {
        'predictions': predictions,
        'probabilities': probabilities,
        'confidence_scores': confidence_scores,
        'average_confidence': average_confidence
    }

print("✅ Simple prediction function loaded!")

# =============================================================================
# STEP 4: COMPLETE EXAMPLE - COPY THIS TO YOUR DATABRICKS!
# =============================================================================

def complete_example():
    """
    📚 COMPLETE WORKING EXAMPLE
    Copy this code and replace the data loading part with your actual data!
    """
    
    print("🎯 COMPLETE ORDINAL CLASSIFICATION EXAMPLE")
    print("=" * 50)
    
    # =======================================================
    # REPLACE THIS SECTION WITH YOUR ACTUAL DATA LOADING!
    # =======================================================
    
    # Your code should look like this:
    # X_train = your training features
    # X_test = your test features  
    # y_train = your training labels (must be 0,1,2,3,4,5,6)
    # y_test = your test labels (must be 0,1,2,3,4,5,6)
    
    # Check your data format:
    print("📋 Data Format Check:")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_test shape: {X_test.shape}")
    print(f"   y_train shape: {y_train.shape}")  
    print(f"   y_test shape: {y_test.shape}")
    print(f"   y_train classes: {np.unique(y_train)}")
    print(f"   y_test classes: {np.unique(y_test)}")
    print()
    
    # =======================================================
    # TRAIN THE MODEL (ONE SIMPLE FUNCTION CALL!)
    # =======================================================
    
    print("🚀 TRAINING THE MODEL...")
    results = train_simple_ordinal_model(
        X_train, X_test, y_train, y_test,
        quick_mode=True  # Set to False for better results (but slower)
    )
    
    if results is None:
        print("❌ Training failed! Check your data format.")
        return
    
    # =======================================================
    # GET YOUR TRAINED MODEL
    # =======================================================
    
    trained_model = results['model']
    data_scaler = results['scaler']
    
    print(f"✅ Model trained successfully!")
    print(f"   Final accuracy: {results['test_accuracy']:.3f}")
    print(f"   Final MAE: {results['test_mae']:.3f}")
    print()
    
    # =======================================================
    # USE MODEL FOR NEW PREDICTIONS (WHEN YOU HAVE NEW DATA)
    # =======================================================
    
    # Uncomment this when you have new data to predict:
    # 
    # print("🔮 PREDICTING NEW DATA...")
    # new_results = predict_new_data(trained_model, data_scaler, X_new_data)
    # new_predictions = new_results['predictions']
    # new_probabilities = new_results['probabilities']
    
    # =======================================================
    # SAVE YOUR MODEL FOR LATER USE
    # =======================================================
    
    print("💾 SAVING MODEL...")
    try:
        import joblib
        
        # Save both model and scaler together
        model_package = {
            'model': trained_model,
            'scaler': data_scaler,
            'accuracy': results['test_accuracy'],
            'mae': results['test_mae']
        }
        
        joblib.dump(model_package, '/dbfs/simple_ordinal_model.pkl')
        print("   ✅ Model saved to /dbfs/simple_ordinal_model.pkl")
        
    except Exception as e:
        print(f"   ⚠️ Could not save model: {e}")
    
    print("\n🎉 EXAMPLE COMPLETED SUCCESSFULLY!")
    print("🎯 Your ordinal classifier is ready to use!")
    
    return results

# =============================================================================
# HOW TO LOAD SAVED MODEL
# =============================================================================

def load_saved_model(file_path='/dbfs/simple_ordinal_model.pkl'):
    """
    📂 LOAD YOUR SAVED MODEL
    """
    try:
        import joblib
        model_package = joblib.load(file_path)
        
        print("✅ Model loaded successfully!")
        print(f"   Accuracy: {model_package['accuracy']:.3f}")
        print(f"   MAE: {model_package['mae']:.3f}")
        
        return model_package['model'], model_package['scaler']
    
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        return None, None

print("✅ Complete example code loaded!")

# =============================================================================
# SUMMARY FOR BEGINNERS
# =============================================================================

print("\n" + "="*80)
print("🎓 SUMMARY FOR BEGINNERS")
print("="*80)
print("📚 WHAT THIS CODE DOES:")
print("   1. Takes your 7-class problem (0,1,2,3,4,5,6)")
print("   2. Converts it into 6 simple YES/NO questions")
print("   3. Trains 6 smart machines to answer each question") 
print("   4. Combines answers to predict the final class")
print("   5. Keeps the natural order: 0 < 1 < 2 < 3 < 4 < 5 < 6")
print()
print("🔑 KEY CONCEPTS:")
print("   • Frank & Hall Method = Convert ordinal to binary questions")
print("   • XGBoost = Smart decision tree algorithm")
print("   • Grid Search = Automatically find best settings")
print("   • Cross-validation = Test settings multiple times")
print("   • Regularization = Prevent overfitting (memorizing)")
print("   • Probability normalization = Make probabilities sum to 1")
print()
print("🚀 TO USE IN DATABRICKS:")
print("   1. Copy this entire code to one cell")
print("   2. Load your data into X_train, X_test, y_train, y_test")
print("   3. Make sure y_train and y_test contain only 0,1,2,3,4,5,6")
print("   4. Run: results = complete_example()")
print("   5. Your trained model is ready!")
print("="*80)

print("\n✅ BEGINNER-FRIENDLY ORDINAL CLASSIFIER READY!")
print("🎯 Everything you need in one simple file!")
print("🚀 Just load your data and run complete_example()!")