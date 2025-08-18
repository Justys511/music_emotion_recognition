
Emotify – Detekcia emócií z hudby

📝 Návod na spustenie projektu

🔧 Požiadavky
- Python 3.10 alebo vyšší  
- Virtuálne prostredie (venv) pre backend  
- Node.js a npm pre frontend (globálne nainštalované v systéme)

---

📁 1. Spustenie backendu (cez venv)

a) Prejdi do koreňovej zložky:

cd Emotion_Detection/


b) Vytvor virtuálne prostredie:

python -m venv venv


c) Aktivuj venv:
- Windows:

venv\Scripts\activate

- Linux/macOS:

source venv/bin/activate


d) Nainštaluj Python závislosti:

pip install -r requirements.txt


e) Spusť backend (API):

cd backend
uvicorn api:app --reload


---

🌐 2. Spustenie frontendu (v samostatnom termináli)

a) Prejdi do priečinka s frontendom:

cd emotion-ui


b) Nainštaluj npm balíky:

npm install


c) Spusť aplikáciu:

npm start


Frontend sa otvorí automaticky v prehliadači na adrese:  
`http://localhost:3000`

---

🎯 Výsledok

Aplikácia umožňuje nahrať hudobný súbor a vizualizovať rozpoznané emócie podľa škály GEMS. Backend klasifikuje zvuk a frontend zobrazuje výsledky.
