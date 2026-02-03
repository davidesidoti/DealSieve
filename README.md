# DealSieve
Filtra automaticamente i messaggi dei gruppi Telegram di offerte e inoltra solo quelli che matchano i prodotti/interessi che configuri.

## Requisiti
- Python 3.10+
- Un account Telegram con accesso ai gruppi pubblici che vuoi filtrare
- API Telegram (api_id, api_hash) create su https://my.telegram.org

## Setup rapido
1. Crea le credenziali API su https://my.telegram.org (API development tools).
2. Copia il file di configurazione:
   - `copy config.example.yaml config.yaml`
3. Compila `config.yaml` con le tue credenziali e i gruppi da filtrare.
4. Installa le dipendenze:
   - `pip install -r requirements.txt`
5. Avvia il bot (al primo run ti chiede numero e codice Telegram):
   - `python app.py`

## Note sulla configurazione
- `filters.groups_allowlist`: lista di username o ID dei gruppi pubblici da monitorare.
- `filters.keywords_include`: almeno una parola chiave deve comparire nel testo.
- `filters.keyword_match`: `whole` per match a parola intera (default), `substring` per match parziale.
- `filters.require_store_link`: se true ignora i messaggi senza link ammessi.
- `filters.allowed_link_domains`: domini/shortener ammessi (Amazon/Aliexpress, ecc.).
- `filters.require_amazon_link`: retrocompatibile se presente, ma preferisci `require_store_link`.
- `logging.color`: abilita colori ANSI nei log (true/false).
- `dedupe.ttl_hours`: finestra di deduplica in ore (default 24).
- `output.send_to`: usa `me` per ricevere i DM sul tuo account.
- `output.mode`: `telethon` (default) o `bot_api`.
- `output.bot_token` e `output.bot_chat_id`: se usi `bot_api`, imposta il token BotFather e il tuo user id.
- `telegram.session_path`: directory dove salvare il file `.session` (opzionale).

## Avvio su VPS (consigliato)
Esegui `python app.py` in una sessione screen/tmux o crea un servizio systemd.

## Risoluzione problemi
- `database is locked`: assicurati che non ci siano altre istanze del bot attive. In alternativa, elimina i file `.session` della sessione o cambia `telegram.session_name`.
- Per eliminare la sessione attuale: `python app.py --clear-session`

## Notifiche con alert (consigliato)
I messaggi verso `me` non generano notifiche (sono uscita dal tuo stesso account). Se vuoi alert sonori/visivi:
1. Crea un bot con @BotFather e prendi il token.
2. Avvia una chat con il bot e recupera il tuo `user id` (es. @userinfobot).
3. Imposta in `config.yaml`:
   - `output.mode: "bot_api"`
   - `output.bot_token: "<TOKEN>"`
   - `output.bot_chat_id: <TUO_USER_ID>`
