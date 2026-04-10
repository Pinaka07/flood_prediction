from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.configuration.config import TEST_SIZE,RANDOM_STATE,MODEL_DIR
import joblib,os

def split_scale(X,y):

    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=TEST_SIZE,random_state=RANDOM_STATE,stratify=y)

    sc=StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.transform(X_test)

    os.makedirs(MODEL_DIR,exist_ok=True)
    joblib.dump(sc,os.path.join(MODEL_DIR,"scaler.pkl"))

    return X_train,X_test,y_train,y_test,sc