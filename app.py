import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pickle
import os
import subprocess
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import json

# Declaring the teams
teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

# declaring the venues
cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

st.markdown("""
<style>
/* Page background: Midnight Purple (subtle, elegant) */
.stApp, .main, .block-container {
  background: linear-gradient(180deg, #0f172a 0%, #2a1640 45%, #4f46e5 100%);
  color: #efeefe;
  min-height: 100vh;
}

/* Title style */
h1 {
  color: #fff;
  text-shadow: 0 4px 14px rgba(46, 12, 78, 0.22);
  font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}

/* Inputs: number, text, textarea, and select */
input[type="number"], input[type="text"], textarea, select {
  color: #ffffff;                                  /* white text for visibility */
  background: rgba(255,255,255,0.03);              /* subtle light overlay on dark bg */
  border: 1px solid rgba(255,255,255,0.12);        /* subtle border */
  border-radius: 8px;
  padding: 6px 8px;
  box-shadow: none;
  font-weight: 600;
}

/* Placeholder color */
input::placeholder, textarea::placeholder {
  color: rgba(255,255,255,0.7);
}

/* Focus state (neon glow) */
input[type="number"]:focus, input[type="text"]:focus, select:focus, textarea:focus {
  outline: none;
  box-shadow: 0 0 18px 6px rgba(124,58,237,0.18), 0 0 6px 2px rgba(96,165,250,0.06);
  border-color: rgba(124,58,237,0.9);
}

/* Invalid values (red) */
input[type="number"]:invalid, input[aria-invalid="true"] {
  color: #ffffff;                                   /* white text on red bg to remain readable */
  background: rgba(255,99,71,0.12);
  border-color: #ff6b6b;
  box-shadow: 0 0 0 6px rgba(255,99,71,0.12);
}

/* Select boxes */
div[data-baseweb="select"] > div {
  background: rgba(255,255,255,0.03) !important;
  color: #ffffff !important;
}


/* Buttons: neon gradient + glow */
.stButton>button, button {
  background: linear-gradient(90deg,#7c3aed,#ff66b3);
  color: #fff;
  border-radius: 12px;
  border: none;
  padding: 10px 16px;
  font-weight: 800;
  box-shadow: 0 8px 30px rgba(124,58,237,0.28);
}

/* Headers for prediction results */
.prediction-header {
  background: linear-gradient(90deg, rgba(124,58,237,0.12), rgba(255,102,179,0.06));
  padding: 8px 10px;
  border-radius: 6px;
  display: inline-block;
  color: #fff;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">IPL Win Predictor</h1>', unsafe_allow_html=True)

# Try to load trained pipeline (pipe.pkl). If missing, allow user to train from the notebook or use fallback training.
pipe = None
if os.path.exists('pipe.pkl'):
    try:
        pipe = pickle.load(open('pipe.pkl', 'rb'))
    except Exception as e:
        st.warning(f'Could not load pipe.pkl: {e}')
        pipe = None


def fallback_train_and_save(path='pipe.pkl'):
    """Fallback training using RandomForest on synthetic data (if notebook training can't run).
       This produces a simple pipeline and saves it to `path`.
    """
    st.info('Running fallback RandomForest training...')
    # Create synthetic dataset
    n = 5000
    rng = np.random.RandomState(42)

    batting = rng.choice(teams, size=n)
    bowling = rng.choice(teams, size=n)
    city = rng.choice(cities, size=n)
    runs_left = rng.randint(0, 200, size=n)
    balls_left = rng.randint(0, 120, size=n)
    wickets_left = rng.randint(0, 10, size=n)
    total_runs = runs_left + rng.randint(50, 250, size=n)
    cur_run_rate = rng.rand(n) * 10
    req_run_rate = (runs_left * 6) / np.where(balls_left == 0, 1, balls_left)

    # heuristic label: batting side wins if required run rate < cur_run_rate + small margin
    labels = (req_run_rate < (cur_run_rate + rng.rand(n) * 1.5)).astype(int)

    df = pd.DataFrame({
        'batting_team': batting,
        'bowling_team': bowling,
        'city': city,
        'runs_left': runs_left,
        'balls_left': balls_left,
        'wickets': wickets_left,
        'total_runs_x': total_runs,
        'cur_run_rate': cur_run_rate,
        'req_run_rate': req_run_rate
    })

    cat_features = ['batting_team', 'bowling_team', 'city']
    num_features = ['runs_left', 'balls_left', 'wickets', 'total_runs_x', 'cur_run_rate', 'req_run_rate']

    preproc = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
    ], remainder='passthrough')

    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    pipeline = Pipeline(steps=[('pre', preproc), ('clf', clf)])

    pipeline.fit(df[cat_features + num_features], labels)

    pickle.dump(pipeline, open(path, 'wb'))
    st.success('Fallback model trained and saved as pipe.pkl')
    return pipeline


def train_from_notebook(nb_name='IPL Win Probability Predictor.ipynb'):
    """Attempt to execute the project's training notebook to create `pipe.pkl`.
       Falls back to `fallback_train_and_save` on failure.
    """
    st.info(f'Executing notebook `{nb_name}` to train model. This may take a while...')
    try:
        cmd = [sys.executable, '-m', 'nbconvert', '--to', 'notebook', '--execute', nb_name,
               '--ExecutePreprocessor.timeout=1200', '--output', 'executed_training.ipynb']
        subprocess.run(cmd, check=True)
        if os.path.exists('pipe.pkl'):
            st.success('Training notebook executed and `pipe.pkl` created.')
            return pickle.load(open('pipe.pkl', 'rb'))
        else:
            st.warning('Notebook executed but `pipe.pkl` was not created. Falling back to synthetic training.')
            return fallback_train_and_save()
    except subprocess.CalledProcessError as e:
        st.error(f'Notebook execution failed: {e}. Falling back to synthetic training.')
        return fallback_train_and_save()


col1, col2 = st.columns(2)
with col1:
    battingteam = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowlingteam = st.selectbox('Select the bowling team', sorted(teams))

city = st.selectbox('Select the city where the match is being played', sorted(cities))

# Inputs with validation constraints
target = st.number_input('Target (runs to chase) — set 0 for first innings', min_value=0, step=1, format='%d')
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Score (current)', min_value=0, step=1, format='%d')
with col4:
    overs = st.number_input('Overs Completed ', min_value=0.0, max_value=20.0, step=0.1, format='%.1f')
with col5:
    wickets_fallen = st.number_input('Wickets Fallen', min_value=0, max_value=10, step=1, format='%d')

# Training UI when model missing
if pipe is None:
    st.warning('Model not found: `pipe.pkl` is missing.')
    if st.button('Train model now (runs the notebook)'):
        with st.spinner('Training...'):
            pipe = train_from_notebook()


if st.button('Predict Probability'):
    # Validation checks
    if battingteam == bowlingteam:
        st.error('Please select two different teams for batting and bowling.')
        st.stop()

    if score < 0 or target < 0:
        st.error('Target must be >= 0 and Score must be >= 0.')
        st.stop()

    # If target is set to 0, it's the first innings (no chase yet)
    if target == 0:
        st.info('Target not set — this is the first innings. Let the first innings begin.')
        st.stop()

    if target != 0 and score > target:
        st.error('Current score cannot be greater than the target.')
        st.stop()

    # Validate overs: fractional part denotes balls (0-5). e.g., 5.3 -> 5 overs and 3 balls
    if overs < 0 or overs > 20:
        st.error('Overs must be between 0.0 and 20.0')
        st.stop()

    # parse overs into overs and balls using one decimal place
    overs_str = f"{overs:.1f}"
    try:
        overs_int_str, balls_str = overs_str.split('.')
        overs_int = int(overs_int_str)
        balls_part = int(balls_str)
    except Exception:
        st.error('Invalid overs format. Use one decimal place, e.g., 5.3 for 5 overs and 3 balls.')
        st.stop()

    # Allow fractional part up to 6; interpret .6 as next over (5.6 -> 6.0)
    if balls_part < 0 or balls_part > 6:
        st.error('Overs fractional part must be number of balls between 0 and 6 (e.g., 5.3; .6 will be converted to next over).')
        st.stop()

    # convert .6 to next over
    if balls_part == 6:
        overs_int += 1
        balls_part = 0
        st.info(f'Input like .6 is treated as next over; interpreted overs as {overs_int}.0')

    if overs_int > 20 or (overs_int == 20 and balls_part > 0):
        st.error('Overs cannot exceed 20.0')
        st.stop()

    total_balls_completed = overs_int * 6 + balls_part
    if total_balls_completed > 120:
        st.error('Total balls completed cannot exceed 120 (20 overs).')
        st.stop()

    # If no balls have been bowled yet, prompt that the match is about to start
    if total_balls_completed == 0:
        st.info('Let the match start — 0.0 overs completed (match not yet started).')
        st.stop()

    if wickets_fallen < 0 or wickets_fallen > 10:
        st.error('Wickets must be between 0 and 10')
        st.stop()

    # Compute derived values safely
    balls_left = int(max(0, 120 - total_balls_completed))
    runs_left = int(max(0, target - score))
    wickets_left = int(max(0, 10 - wickets_fallen))
    currentrunrate = float(score / (overs_int + (balls_part/6))) if total_balls_completed > 0 else 0.0
    requiredrunrate = float((runs_left * 6) / balls_left) if balls_left > 0 else float('inf')

    input_df = pd.DataFrame({'batting_team': [battingteam], 'bowling_team': [bowlingteam], 'city': [city],
                             'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets': [wickets_left],
                             'total_runs_x': [target], 'cur_run_rate': [currentrunrate], 'req_run_rate': [requiredrunrate]})

    if pipe is None:
        st.error('No trained model available. Please train the model first.')
        st.stop()

    try:
        result = pipe.predict_proba(input_df)
        lossprob = result[0][0]
        winprob = result[0][1]

        # Determine winner and loser
        if winprob >= lossprob:
            winner_name = battingteam
            winner_prob = winprob
            loser_name = bowlingteam
            loser_prob = lossprob
        else:
            winner_name = bowlingteam
            winner_prob = lossprob
            loser_name = battingteam
            loser_prob = winprob

        # Display winner first with a highlighted header, loser below
        st.markdown(f"<div class=\"prediction-header\" style=\"padding:12px 14px; border-radius:10px; font-size:20px;\">"
                    f"<strong>{winner_name}</strong> — <span style='color:#7efc82;'>{round(winner_prob*100)}%</span> "
                    f"<small style='opacity:0.85; margin-left:8px;'>Likely to win</small></div>", unsafe_allow_html=True)

        st.markdown(f"<div style=\"margin-top:8px; color:rgba(255,255,255,0.85); font-size:16px;\">{loser_name} — <span style='color:#ff9a9a;'>{round(loser_prob*100)}%</span></div>", unsafe_allow_html=True)

        # If close match, show a note
        if abs(winner_prob - loser_prob) < 0.10:
            st.info('This is a close contest — probabilities are within 10%.')

        # Celebration: Only for confident predictions (threshold)
        celebration_threshold = 0.60
        if winner_prob >= celebration_threshold:
            try:
                # Show a congratulatory banner + fireworks confetti (customized)
                colors = ['#7C3AED','#FF66B3','#7EE7C2','#FFD166']
                # stronger celebration for very confident predictions
                if winner_prob >= 0.85:
                    duration = 3500
                else:
                    duration = 2500
                html_template = """
<div style='text-align:center; margin-top:8px;'>
  <div style='display:inline-block; padding:12px 18px; border-radius:12px; background: linear-gradient(90deg,#4f46e5,#a78bfa); color:white; font-weight:800; box-shadow: 0 8px 20px rgba(79,70,229,0.18);'>
    Congratulations — {winner} <span style='opacity:0.9; margin-left:8px;'> {winner_prob}% </span>
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
<script>
(function() {{
  var duration = {duration};
  var animationEnd = Date.now() + duration;
  var defaults = {{ startVelocity: 30, spread: 360, ticks: 60, zIndex: 2000 }};
  var colors = {colors};
  var interval = setInterval(function() {{
    var timeLeft = animationEnd - Date.now();
    if (timeLeft <= 0) {{
      return clearInterval(interval);
    }}
    var particleCount = Math.floor(50 * (timeLeft / duration));
    confetti(Object.assign({{}}, defaults, {{ particleCount: particleCount, colors: colors, origin: {{ x: Math.random(), y: Math.random()*0.6 }} }}));
  }}, 250);
}})();
</script>
"""
                html = html_template.format(duration=duration, colors=json.dumps(colors), winner=winner_name, winner_prob=round(winner_prob*100))
                components.html(html, height=360)
            except Exception:
                pass
    except Exception as e:
        st.error(f'Prediction failed: {e}')
        st.stop()
