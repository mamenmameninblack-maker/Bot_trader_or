import time
from datetime import datetime

from analyse_or import analyze_xauusd
from ctrader_api import get_api, is_in_cooldown, activate_cooldown

# === VARIABLES GLOBALES ===
trades_today = 0
current_day = datetime.utcnow().date()
loss_streak = 0


# ============================================================
# BOT TRADER OR — STRUCTURE PRINCIPALE
# ============================================================

def run_single_cycle():
    global trades_today, current_day, loss_streak

    # Reset si nouveau jour
    if datetime.utcnow().date() != current_day:
        trades_today = 0
        loss_streak = 0
        run_single_cycle.starting_equity = None  # reset du capital du jour
        current_day = datetime.utcnow().date()

    print("\n======================")
    print("   BOT TRADER OR")
    print("======================")
    print(f"Cycle lancé à : {datetime.utcnow()} UTC\n")

    # 1) Analyse complète
    resultat = analyze_xauusd()

    signal = resultat.get("signal")
    last_price = resultat.get("last_price")
    atr_value = resultat.get("atr_value")

    # 🔥 1) Cooldown actif ?
    if is_in_cooldown():
        print("⛔ Cooldown actif — aucun trade autorisé.")
        return

    # 🔥 2) Cooldown volatilité extrême ?
    if signal == "COOLDOWN_VOL":
        activate_cooldown(hours=0.5)
        print("⛔ Cooldown volatilité extrême activé pour 30 minutes.")
        return

    print("Signal :", signal)
    print("Prix :", last_price)
    print("ATR :", atr_value)

    # 2) Vérifications minimales
    if signal == "NEUTRE":
        print("→ Signal neutre, aucun ordre envoyé.")
        return

    if last_price is None or atr_value is None:
        print("→ Données insuffisantes pour trader.")
        return

    # === STOP SI DRAWDOWN JOURNALIER (30%) ===
    api = get_api()
    account = api.get_account_info()
    equity = float(account.get("equity", 0))

    daily_dd_max = 0.30  # 30% de perte max dans la journée

    # Initialisation du capital du jour
    if not hasattr(run_single_cycle, "starting_equity") or run_single_cycle.starting_equity is None:
        run_single_cycle.starting_equity = equity

    # Stop si drawdown journalier atteint
    if equity < run_single_cycle.starting_equity * (1 - daily_dd_max):
        print("⛔ Drawdown journalier de 30% atteint — arrêt du trading jusqu’à demain.")
        return

    # === LIMITE TRADES PAR JOUR ===
    if trades_today >= 15:
        print("⛔ Max trades atteint aujourd’hui (15).")
        return

    # 3) Exécution du trade
    exec_res = api.execute_signal_with_atr(
        symbol="XAUUSD",
        signal=signal,
        last_price=last_price,
        atr_value=atr_value,
    )

    print("\nRésultat exécution ordre :")
    print(exec_res)

    # Incrémenter le compteur de trades
    trades_today += 1

    # 4) Vérifier si une position s'est fermée
    closed_positions = api.get_closed_positions()

    if closed_positions:
        last_closed = closed_positions[-1]
        api.notify_closed_position(last_closed)

        # === GESTION DES PERTES CONSÉCUTIVES ===
        profit = last_closed.get("profit")

        try:
            profit = float(profit)
        except:
            profit = None

        if profit is not None:
            if profit < 0:
                loss_streak += 1
            else:
                loss_streak = 0

        if loss_streak >= 4:
            print("❌ 4 pertes consécutives — arrêt du trading jusqu’à demain.")
            return


# ============================================================
# BOUCLE AUTOMATIQUE (VERSION PRO)
# ============================================================

def run_bot_loop(interval_seconds=60):
    print("\n==============================")
    print("   BOT TRADER OR — MODE AUTO")
    print("==============================\n")

    cycle = 0

    while True:
        cycle += 1
        print(f"\n===== CYCLE {cycle} =====")

        try:
            run_single_cycle()
        except Exception as e:
            print("\n[ERREUR] Le bot a rencontré une exception :", e)
            print("Le bot continue automatiquement...\n")

        print(f"⏳ Prochain cycle dans {interval_seconds} secondes...\n")
        time.sleep(interval_seconds)


# ============================================================
# LANCEUR FINAL
# ============================================================

if __name__ == "__main__":
    print("Démarrage du bot trader OR...\n")
    run_bot_loop(interval_seconds=60)
