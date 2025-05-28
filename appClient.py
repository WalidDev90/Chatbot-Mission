from flask import Flask, request, jsonify, render_template, send_from_directory, session
from charset_normalizer import detect
import pandas as pd
import google.generativeai as genai
from langchain.agents import initialize_agent, Tool
from langchain.llms import GooglePalm
from langchain.agents.agent_types import AgentType
import os
import pickle
from langchain_experimental.tools.python.tool import PythonREPLTool  # ✅
from langchain_google_genai import ChatGoogleGenerativeAI
import traceback
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import re
from datetime import datetime
import io
import contextlib

# Configure Gemini (via Google Generative AI)
genai.configure(api_key="AIzaSyAjIBfty23aEyZ7rLfxa171JrejzLL1uGc")  # Remplace par ta clé API Google

# === CHARGEMENT DES DATASETS ===
with open("Dataset_Banking_chatbot.csv", 'rb') as f:
    result = detect(f.read())
encoding = result['encoding']

banking_kaggle  = pd.read_csv("Dataset_Banking_chatbot.csv", encoding=encoding)
banking_kaggle1 = pd.read_csv("banking-chatbot-dataset.csv", encoding=encoding)
banking_kaggle2 = pd.read_csv("cards_data.csv", encoding=encoding)
banking_kaggle3 = pd.read_csv("users_data.csv", encoding=encoding)

# === DONNÉES ===
faq_df = banking_kaggle

df1 = banking_kaggle1
df1['Date'] = pd.to_datetime(df1['Date'], format='%d/%m/%y')

df2 = banking_kaggle2
df2.loc[:, "credit_limit"] = df2["credit_limit"].replace('[\$,]', '', regex=True).astype(float)

df3 = banking_kaggle3
df3['is_retired'] = df3['current_age'] >= df3['retirement_age']
for col in ['per_capita_income', 'yearly_income', 'total_debt']:
    df3.loc[:, col] = df3[col].replace('[\$,]', '', regex=True).astype(float)
current_year = datetime.now().year
df3.loc[:, 'current_age'] = current_year - df3['birth_year']

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Préparer les embeddings des questions dans la FAQ (à faire 1 seule fois)
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
faq_queries = faq_df['Query'].tolist()
faq_embeddings = model.encode(faq_queries, convert_to_tensor=True)

# === OUTILS PERSONNALISÉS ===
def detect_langue(question):
    prompt = f"What'is the language of question response in one word:\n\n'{question}'"
    m = genai.GenerativeModel("gemini-1.5-flash")
    langue = m.generate_content(prompt).text.strip().lower()
    return langue

def traduire_reponse_langue(reponse, langue):
    prompt = f"Traduire la phrase suivante en {langue} (short and natural in one sentence):\n\n'{reponse}'"
    m = genai.GenerativeModel("gemini-1.5-flash")
    return m.generate_content(prompt).text.strip().lower()

def traduire(question):
    prompt = f"Translate this user question to English (short and natural in one sentence):\n\n'{question}'"
    m = genai.GenerativeModel("gemini-1.5-flash")
    return m.generate_content(prompt).text.strip().lower()

def chercher_faq(question):
    langue = detect_langue(question)
    if langue != "english":
        question = traduire(question)
    question_embedding = model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, faq_embeddings)[0]
    best_idx = scores.argmax().item()
    if scores[best_idx] > 0.6:
        resp = faq_df.iloc[best_idx]['Response']
        if langue != "english":
            resp = traduire_reponse_langue(resp, langue)
        return resp
    else:
        return "Je n'ai pas trouvé de réponse proche dans la FAQ."

def interroger_dataframe(question: str) -> str:
    prompt = f"""
Tu es un assistant expert en analyse de données financières.

Voici la question de l'utilisateur :
>>> {question}

Tu dois générer uniquement du code Python, SANS recréer les DataFrames. Utilise uniquement ceux qui existent déjà dans l’environnement :

- df1 : colonnes = {list(df1.columns)} qui représente mon relevé bancaire

📌 Consignes strictes :
- ❌ Ne crée aucun DataFrame.
- ✅ Utilise uniquement df1.
- ✅ Fournis uniquement le code Python final, encadré par ```python ...```.
- ✅ Termine toujours par un print clair du résultat.
"""
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        generated = response.content.strip()
        match = re.search(r"```python(.*?)```", generated, re.DOTALL)
        if not match:
            return "❌ Aucun code Python détecté."
        code = match.group(1).strip()
        local_vars = {"df1": df1}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(code, {}, local_vars)
        return output.getvalue().strip()
    except Exception as e:
        return f"❌ Erreur pendant l’exécution : {e}"

# Créer les outils LangChain
faq_tool = Tool(
    name="FAQ Tool",
    func=chercher_faq,
    description="Utilise la FAQ pour répondre aux questions textuelles connues"
)
data_tool = Tool(
    name="Data Tool",
    func=interroger_dataframe,
    description="Utilise les données df1 pour répondre aux questions chiffrées"
)
python_tool = PythonREPLTool()

# === AGENT IA ===
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key="AIzaSyAjIBfty23aEyZ7rLfxa171JrejzLL1uGc")
agent = initialize_agent(
    tools=[faq_tool, data_tool, python_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# === ROUTES FLASK ===

@app.route('/')
def index():
    # Utilisateur connecté = premier utilisateur réel de la base
    user_row = df3.iloc[0]
    user = {
        'name': user_row['name'],
        'balance': float(user_row['yearly_income']),  # Utilisation de yearly_income comme solde
        'profile_img': user_row['image'] if 'image' in user_row else 'https://i.pravatar.cc/150?img=3',
        'id': user_row['id'] if 'id' in user_row else ''
    }

    # Charger toutes les opérations depuis df1
    operations = df1.sort_values('Date', ascending=True)
    operations_list = []
    for _, row in operations.iterrows():
        operations_list.append({
            'date': row['Date'].strftime('%Y-%m-%d') if not pd.isnull(row['Date']) else '',
            'day': row['Day'] if 'Day' in row and pd.notnull(row['Day']) else (row['Date'].strftime('%A') if not pd.isnull(row['Date']) else ''),
            'type': row.get('Type', ''),
            'category': row.get('Category', ''),
            'debit': float(row.get('Debit Amount', 0) or 0),
            'credit': float(row.get('Credit Amount', 0) or 0),
            'balance': float(row.get('Closing Balance', 0) or 0)
        })

    # Chart.js
    dates    = [op['date']   for op in operations_list]
    balances = [op['balance'] for op in operations_list]
    credits  = [op['credit']  for op in operations_list]
    debits   = [op['debit']   for op in operations_list]

    chart_data = {
        'labels': dates,
        'datasets': [
            {'label': 'Solde',  'data': balances},
            {'label': 'Crédit', 'data': credits},
            {'label': 'Débit',  'data': debits}
        ]
    }
    return render_template(
        'index.html',
        user=user,
        operations=operations_list,
        chart_data=chart_data
    )

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400

    if 'history' not in session:
        session['history'] = []
    history = session['history']

    client_id_match = re.search(r'(?:client|numéro|numero)[^\d]*(\d+)', user_input, re.IGNORECASE)
    if client_id_match:
        session['client_id'] = client_id_match.group(1)
    client_id = session.get('client_id')
    if client_id and re.search(r'\b(le|la) client\b', user_input, re.IGNORECASE) and not re.search(r'\d+', user_input):
        user_input = re.sub(r'\b(le|la) client\b', f'le client {client_id}', user_input, flags=re.IGNORECASE)

    langue = detect_langue(user_input)
    question_trad = user_input if langue == 'english' else traduire(user_input)
    history.append({'role': 'user', 'content': question_trad})

    try:
        reponse = agent.llm.invoke(history).content.strip()
    except Exception:
        reponse = agent.run(question_trad)

    history.append({'role': 'assistant', 'content': reponse})
    session['history'] = history

    if langue != 'english':
        reponse = traduire_reponse_langue(reponse, langue)

    return jsonify({'response': reponse.capitalize()})

@app.route('/filter', methods=['POST'])
def filter_operations():
    data = request.get_json()
    start = data.get('start')
    end   = data.get('end')
    op_type = data.get('type')
    filtered = df1.copy()
    if start:
        filtered = filtered[filtered['Date'] >= pd.to_datetime(start)]
    if end:
        filtered = filtered[filtered['Date'] <= pd.to_datetime(end)]
    if op_type and op_type.lower() != 'tous':
        filtered = filtered[filtered['Type'].str.lower() == op_type.lower()]
    filtered = filtered.sort_values('Date', ascending=True)

    operations_list = []
    for _, row in filtered.iterrows():
        operations_list.append({
            'date': row['Date'].strftime('%Y-%m-%d') if not pd.isnull(row['Date']) else '',
            'day': row['Day'] if 'Day' in row and pd.notnull(row['Day']) else (row['Date'].strftime('%A') if not pd.isnull(row['Date']) else ''),
            'type': row.get('Type', ''),
            'category': row.get('Category', ''),
            'debit': float(row.get('Debit Amount', 0) or 0),
            'credit': float(row.get('Credit Amount', 0) or 0),
            'balance': float(row.get('Closing Balance', 0) or 0)
        })
    return jsonify({'operations': operations_list})

@app.route('/transfer', methods=['POST'])
def transfer():
    data = request.get_json()
    from_user = data.get('fromUser')
    to_user = data.get('toUser')
    amount = float(data.get('amount', 0))
    if not from_user or not to_user or amount <= 0:
        return jsonify({'success': False, 'error': 'Champs invalides.'})

    def find_user(query):
        # Recherche par ID numérique
        if str(query).isdigit():
            row = df3[df3['id'] == int(query)]
            if not row.empty:
                return row
        # Recherche par email si colonne présente
        if 'email' in df3.columns:
            row = df3[df3['email'].str.lower() == str(query).strip().lower()]
            if not row.empty:
                return row
        # Recherche par nom/prénom (tolérance accents, espaces, casse)
        import unicodedata
        def normalize(s):
            return ''.join(c for c in unicodedata.normalize('NFD', str(s).lower()) if c.isalnum())
        qnorm = normalize(query)
        for idx, row in df3.iterrows():
            if normalize(row['name']) == qnorm:
                return df3.loc[[idx]]
            if 'first_name' in df3.columns and 'last_name' in df3.columns:
                if normalize(row['first_name']) == qnorm or normalize(row['last_name']) == qnorm:
                    return df3.loc[[idx]]
        return pd.DataFrame()

    from_row = find_user(from_user)
    to_row = find_user(to_user)
    if from_row.empty or to_row.empty:
        return jsonify({'success': False, 'error': "Utilisateur introuvable."})
    from_idx = from_row.index[0]
    to_idx = to_row.index[0]
    # Utilisation de yearly_income comme solde
    if df3.at[from_idx, 'yearly_income'] < amount:
        return jsonify({'success': False, 'error': "Solde insuffisant."})
    df3.at[from_idx, 'yearly_income'] -= amount
    df3.at[to_idx, 'yearly_income'] += amount
    return jsonify({'success': True, 'new_balance': round(df3.at[from_idx, 'yearly_income'], 2)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
