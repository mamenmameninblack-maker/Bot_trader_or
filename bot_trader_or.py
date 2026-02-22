import time
from datetime import datetime

from analyse_or import analyze_xauusd
from ctrader_api import get_api

# ============================================================
# BOT TRADER OR — STRUCTURE PRINCIPALE
# ============================================================

def run_single_cycle():
    """
    Exécute un cycle complet :
    - Analyse XAUUSD
    - Exécution du signal (ACHAT / VENTE / NEUTRE)
    - Affichage du résultat
    """

    print("\n======================")
    print("   BOT TRADER OR")
    print("======================")
    print(f"Cycle lancé à : {datetime.utcnow()} UTC\n")

    # 1) Analyse complète
    resultat = analyze_xauusd()

    signal = resultat.get("signal")
    last_price = resultat.get("last_price")
    atr_value = resultat.get("atr_value")

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

    # 3) Exécution du trade
    api = get_api()

    exec_res = api.execute_signal_with_atr(
        symbol="XAUUSD",
        signal=signal,
        last_price=last_price,
        atr_value=atr_value,
    )

    print("\nRésultat exécution ordre :")
    print(exec_res)

    # 4) Vérifier si une position s'est fermée
    closed_positions = api.get_closed_positions()

    if closed_positions:
        last_closed = closed_positions[-1]  # dernière position fermée
        api.notify_closed_position(last_closed)


# ============================================================
# BOUCLE AUTOMATIQUE (VERSION PRO)
# ============================================================

def run_bot_loop(interval_seconds=60):
    """
    Boucle automatique :
    - Exécute un cycle complet toutes les X secondes
    - Protège contre les crashs
    - Continue à tourner 24/7
    """

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

    # MODE 1 : un seul cycle (test rapide)
    # run_single_cycle()

    # MODE 2 : boucle automatique (production)
    run_bot_loop(interval_seconds=60)
