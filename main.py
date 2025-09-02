# main.py
# WhatsApp AI Agent – Commandes (R4)
# FastAPI + SQLAlchemy + WhatsApp Business Cloud API v22
# - Menu interactif + fallback texte
# - Parsing robuste ("2 margherita", "2x carbonara", "2 margherita et 1 coca")
# - Réponse aux list replies
# - Flux restaurant (confirmation & statuts) + notifications client
# - Modifications panier (ajouter / supprimer / vider)

import os
import re
import json
import logging
import unicodedata
from datetime import datetime
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship

import requests

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
class Config:
    WHATSAPP_TOKEN: str = os.getenv("WHATSAPP_TOKEN", "your_whatsapp_token")
    WHATSAPP_PHONE_ID: str = os.getenv("WHATSAPP_PHONE_ID", "your_phone_id")
    WHATSAPP_VERIFY_TOKEN: str = os.getenv("WHATSAPP_VERIFY_TOKEN", "verify_token_123")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./whatsapp_orders.db")
    # Numéro WhatsApp du restaurant (E.164 sans +, ex: 33758262447)
    RESTAURANT_PHONE: str = os.getenv("RESTAURANT_PHONE", "33758262447")

config = Config()

# -----------------------------------------------------------------------------
# DB
# -----------------------------------------------------------------------------
engine = create_engine(config.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class OrderStatus:
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PREPARING = "preparing"
    READY = "ready"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

class Customer(Base):
    __tablename__ = "customers"
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String, unique=True, index=True)
    name = Column(String)
    address = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    orders = relationship("Order", back_populates="customer")

class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text)
    price = Column(Float)
    category = Column(String)
    # string "true"/"false" pour compat avec DB existante
    available = Column(String, default="true")

class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"))
    status = Column(String, default=OrderStatus.PENDING)
    total_amount = Column(Float)
    items = Column(Text)   # JSON
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    customer = relationship("Customer", back_populates="orders")

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String, index=True)
    context = Column(Text)  # JSON
    last_interaction = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -----------------------------------------------------------------------------
# Utils / normalisation texte
# -----------------------------------------------------------------------------
def normalize(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.replace("-", " ").strip()
    return s

def format_lines(items: List[Dict]) -> List[str]:
    return [f"• {i['quantity']}× {i['name']} — €{i['price'] * i['quantity']:.2f}" for i in items]

# -----------------------------------------------------------------------------
# WhatsApp Service (v22)
# -----------------------------------------------------------------------------
class WhatsAppService:
    def __init__(self):
        self.token = config.WHATSAPP_TOKEN
        self.phone_id = config.WHATSAPP_PHONE_ID
        self.base_url = f"https://graph.facebook.com/v22.0/{self.phone_id}"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def send_message(self, to: str, message: str) -> bool:
        url = f"{self.base_url}/messages"
        data = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": message},
        }
        try:
            r = requests.post(url, json=data, headers=self._headers(), timeout=15)
            ok = r.status_code in (200, 201)
            if ok:
                logging.info(f"WA text ok: {r.text}")
            else:
                logging.error(f"WA text failed {r.status_code}: {r.text}")
            return ok
        except Exception as e:
            logging.error(f"WA text error: {e}")
            return False

    def send_interactive_menu(self, to: str, products: List[Dict]) -> bool:
        rows = []
        for p in products[:10]:
            title = p.get("name", "Article")
            desc = (p.get("description") or "").strip()
            price = p.get("price")
            if price is not None:
                desc = (desc + (" - " if desc else "")) + f"€{price:.2f}"
            rows.append({
                "id": f"product_{p.get('id', title)}",
                "title": title[:24],
                "description": desc[:72],
            })
        if not rows:
            return False

        url = f"{self.base_url}/messages"
        data = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "interactive",
            "interactive": {
                "type": "list",
                "header": {"type": "text", "text": "🍕 Notre menu"},
                "body": {"text": "Répondez par ex. : 2 margherita, 1 coca"},
                "footer": {"text": "Tapez 'confirmer' pour valider"},
                "action": {
                    "button": "Voir menu",
                    "sections": [{
                        "title": "Nos produits",
                        "rows": rows
                    }]
                }
            }
        }
        try:
            r = requests.post(url, json=data, headers=self._headers(), timeout=15)
            ok = r.status_code in (200, 201)
            if ok:
                logging.info(f"WA interactive ok: {r.text}")
            else:
                logging.error(f"WA interactive failed {r.status_code}: {r.text}")
            return ok
        except Exception as e:
            logging.error(f"WA interactive error: {e}")
            return False

# -----------------------------------------------------------------------------
# Order Service
# -----------------------------------------------------------------------------
class OrderService:
    def __init__(self, db: Session):
        self.db = db

    def get_or_create_customer(self, phone_number: str) -> Customer:
        c = self.db.query(Customer).filter(Customer.phone_number == phone_number).first()
        if not c:
            c = Customer(phone_number=phone_number)
            self.db.add(c)
            self.db.commit()
            self.db.refresh(c)
        return c

    def create_order(self, phone_number: str, items: List[Dict], notes: str = "") -> Order:
        customer = self.get_or_create_customer(phone_number)
        total = sum(item["price"] * item["quantity"] for item in items)
        order = Order(
            customer_id=customer.id,
            total_amount=total,
            items=json.dumps(items),
            notes=notes,
            status=OrderStatus.PENDING,
        )
        self.db.add(order)
        self.db.commit()
        self.db.refresh(order)
        return order

    def get_order(self, order_id: int) -> Optional[Order]:
        return self.db.query(Order).filter(Order.id == order_id).first()

    def set_status(self, order: Order, status: str):
        order.status = status
        order.updated_at = datetime.utcnow()
        # recalc total (au cas où items modifiés)
        try:
            items = json.loads(order.items or "[]")
            order.total_amount = sum(i["price"] * i["quantity"] for i in items)
        except Exception:
            pass
        self.db.commit()
        self.db.refresh(order)

# -----------------------------------------------------------------------------
# Conversation & Parsing
# -----------------------------------------------------------------------------
class ConversationService:
    def __init__(self, db: Session):
        self.db = db
        self.whatsapp = WhatsAppService()
        self.order_service = OrderService(db)

    # ---- context
    def get_conversation_context(self, phone: str) -> Dict:
        conv = self.db.query(Conversation).filter(Conversation.phone_number == phone).first()
        if conv and conv.context:
            return json.loads(conv.context)
        return {"state": "new", "current_order": []}

    def update_conversation_context(self, phone: str, context: Dict):
        conv = self.db.query(Conversation).filter(Conversation.phone_number == phone).first()
        if not conv:
            conv = Conversation(phone_number=phone)
            self.db.add(conv)
        conv.context = json.dumps(context)
        conv.last_interaction = datetime.utcnow()
        self.db.commit()

    # ---- produits
    def _all_available_products(self) -> List[Product]:
        prods = self.db.query(Product).filter(Product.available == "true").all()
        if not prods:
            prods = self.db.query(Product).all()
        return prods

    # ---- synonymes (mapping texte -> produit)
    def _synonyms_map(self) -> Dict[str, Product]:
        m: Dict[str, Product] = {}
        for p in self._all_available_products():
            full = normalize(p.name)
            words = [w for w in full.split() if w]
            if not words:
                continue
            last = words[-1]
            m[full] = p
            m[last] = p
            if "pepperoni" in full:
                m["pizza pepperoni"] = p; m["pepperoni"] = p
            if "margherita" in full:
                m["pizza margherita"] = p; m["margherita"] = p
            if "carbonara" in full:
                m["carbonara"] = p; m["pasta carbonara"] = p; m["pates carbonara"] = p
            if "cesar" in full or "cesar" in last:
                m["salade cesar"] = p; m["cesar"] = p
            if "coca" in full:
                m["coca"] = p; m["coca cola"] = p
            if "eau" in full:
                m["eau"] = p; m["eau minerale"] = p
        return m

    # ---- intent
    def _detect_intent(self, msg: str) -> str:
        m = normalize(msg)
        if any(w in m for w in ("bonjour", "salut", "hello", "coucou")):
            return "greeting"
        if "menu" in m:
            return "menu"
        if any(w in m for w in ("confirmer", "valider")):
            return "confirm"
        if any(w in m for w in ("supprimer", "retirer", "enlever", "remove", "delete", "annuler un article")):
            return "remove"
        if any(w in m for w in ("ajouter", "ajoute", "add", "plus")):
            return "add"
        if "vider" in m or "tout enlever" in m:
            return "clear"
        if self._parse_items(m):
            return "order"
        return "other"

    # ---- parsing
    def _split_phrases(self, m: str) -> List[str]:
        m = re.sub(r"\s*(,|;|\+|\bet\b)\s*", "|", m)
        parts = [p.strip() for p in m.split("|") if p.strip()]
        return parts or [m]

    def _qty_in_text(self, s: str) -> int:
        m = re.search(r"(\d+)\s*(?:x|×)?", s)
        if m:
            return max(1, int(m.group(1)))
        m = re.search(r"(\d+)\s*$", s)
        return max(1, int(m.group(1))) if m else 1

    def _parse_items(self, msg_norm: str) -> List[Dict]:
        syn = self._synonyms_map()
        if not syn:
            return []
        keys = sorted(syn.keys(), key=len, reverse=True)
        items: List[Dict] = []
        for chunk in self._split_phrases(msg_norm):
            qty = self._qty_in_text(chunk)
            chunk_wo_qty = re.sub(r"^\s*\d+\s*(?:x|×)?\s*", "", chunk).strip()
            picked: Optional[Product] = None
            for k in keys:
                if k and k in chunk_wo_qty:
                    picked = syn[k]
                    break
            if picked:
                items.append({"name": picked.name, "price": float(picked.price), "quantity": qty})
        return items

    # ---- helpers panier
    def _add_items_to_context(self, context: Dict, items: List[Dict]) -> None:
        context.setdefault("current_order", [])
        context["current_order"].extend(items)

    def _remove_items_from_context(self, context: Dict, items: List[Dict]) -> int:
        """Enlève les items demandés du panier. Retourne nb d'unités retirées."""
        removed = 0
        cart = context.get("current_order", [])
        # Map rapide nom -> total qty
        want: Dict[str, int] = {}
        for it in items:
            want[it["name"]] = want.get(it["name"], 0) + max(1, int(it.get("quantity", 1)))

        # Parcours et décrémente
        for name, qty_to_remove in want.items():
            i = 0
            while i < len(cart) and qty_to_remove > 0:
                entry = cart[i]
                if entry["name"].lower() == name.lower():
                    take = min(entry["quantity"], qty_to_remove)
                    entry["quantity"] -= take
                    qty_to_remove -= take
                    removed += take
                    if entry["quantity"] <= 0:
                        cart.pop(i)
                        continue
                i += 1
        context["current_order"] = cart
        return removed

    def _cart_response(self, context: Dict, prefix_ok: str, empty_msg: str) -> str:
        cart = context.get("current_order", [])
        if not cart:
            return empty_msg
        total = sum(i["price"] * i["quantity"] for i in cart)
        return (f"{prefix_ok}\n\n📋 *Récapitulatif*:\n" +
                "\n".join(format_lines(cart)) +
                f"\n\n💰 *Total*: €{total:.2f}\n"
                "Tapez *confirmer* pour valider, ou continuez à ajouter/supprimer des articles.")

    # ---- main dialogue
    def process_incoming_message(self, phone: str, message: str) -> str:
        context = self.get_conversation_context(phone)
        intent = self._detect_intent(message)
        logging.info(f"[intent={intent}] from={phone} msg={message!r} ctx={context}")

        if intent == "greeting":
            response = ("🍕 Bonjour! Bienvenue chez Barita Resto.\n"
                        "Tapez *menu* pour voir nos plats ou dites-moi directement votre commande "
                        "(ex: *2 margherita et 1 coca*).")
            context["state"] = "menu_or_order"

        elif intent == "menu":
            products = self._all_available_products()
            products_dict = [{"id": p.id, "name": p.name, "description": p.description, "price": p.price}
                             for p in products]
            ok = self.whatsapp.send_interactive_menu(phone, products_dict)
            if not ok:
                lines = ["🍕 *Notre menu*"]
                for p in products[:10]:
                    lines.append(f"• {p.name} — €{p.price:.2f}")
                lines.append("\nRépondez par ex. : 2 margherita, 1 coca")
                self.whatsapp.send_message(phone, "\n".join(lines))
            response = "📋 Menu envoyé ! Vous pouvez aussi me dire directement ce que vous voulez."
            context["state"] = "menu_shown"

        elif intent in ("order", "add"):
            items = self._parse_items(normalize(message))
            if items:
                self._add_items_to_context(context, items)
                response = self._cart_response(context, "✅ Ajouté à votre commande !",
                                               "Votre panier est vide.")
                context["state"] = "order_building"
            else:
                response = ("Je n'ai pas bien compris les articles. Donnez un format comme : "
                            "*2 margherita et 1 coca*.")

        elif intent == "remove":
            items = self._parse_items(normalize(message))
            if "vider" in normalize(message) or not items:
                # si pas d'item explicite mais commande remove => tenter vider
                if context.get("current_order"):
                    context["current_order"] = []
                    response = "🧺 Panier vidé."
                else:
                    response = "Votre panier est déjà vide."
            else:
                removed = self._remove_items_from_context(context, items)
                if removed > 0:
                    response = self._cart_response(context, "🗑️ Article(s) retiré(s).",
                                                   "Votre panier est vide après suppression.")
                else:
                    response = "Je n'ai pas trouvé ces articles dans votre panier."

        elif intent == "clear":
            context["current_order"] = []
            response = "🧺 Panier vidé."

        elif intent == "confirm":
            cart = context.get("current_order", [])
            if cart:
                order = self.order_service.create_order(phone, cart)
                # Envoi au restaurant
                total = sum(i["price"] * i["quantity"] for i in cart)
                lines = "\n".join(format_lines(cart))
                admin_msg = (f"🍽️ *Nouvelle commande* #{order.id}\n"
                             f"De: {phone}\n\n{lines}\n\n"
                             f"💰 Total: €{total:.2f}\n\n"
                             f"Répondez: *ok {order.id}* / *preparer {order.id}* / "
                             f"*pret {order.id}* / *livre {order.id}* / *annule {order.id}*")
                self.whatsapp.send_message(config.RESTAURANT_PHONE, admin_msg)

                # Réponse au client (attente confirmation resto)
                response = (f"🎉 Commande #{order.id} envoyée au restaurant.\n"
                            "👨‍🍳 Vous recevrez une notification dès que c'est confirmé.")
                context["state"] = "order_pending_restaurant"
                context["current_order"] = []
                context["last_order_id"] = order.id
            else:
                response = "Votre panier est vide. Ajoutez des articles avant de confirmer !"

        else:
            response = ("Je n'ai pas compris. Tapez *menu* pour voir nos options, "
                        "ou envoyez une commande du type *2 margherita et 1 coca*.")

        self.update_conversation_context(phone, context)
        return response

    # ---- interactive replies (list)
    def process_interactive_reply(self, phone: str, list_reply_id: str, title: str) -> str:
        context = self.get_conversation_context(phone)
        product_id = None
        if list_reply_id.startswith("product_"):
            try:
                product_id = int(list_reply_id.split("_", 1)[1])
            except Exception:
                product_id = None

        if product_id is not None:
            p = self.db.query(Product).filter(Product.id == product_id).first()
            if p:
                item = {"name": p.name, "price": float(p.price), "quantity": 1}
                self._add_items_to_context(context, [item])
                response = self._cart_response(context, "✅ Ajouté au panier depuis le menu.",
                                               "Votre panier est vide.")
                context["state"] = "order_building"
                self.update_conversation_context(phone, context)
                return response

        return ("Je n'ai pas pu ajouter cet élément. Réessayez depuis le *menu* "
                "ou envoyez un message du type *1 margherita*.")

# -----------------------------------------------------------------------------
# Admin / Restaurant commands
# -----------------------------------------------------------------------------
def process_admin_command(db: Session, text: str, whatsapp: WhatsAppService) -> Optional[str]:
    """
    Commandes admin (restaurant) par WhatsApp :
      - ok 123         -> confirmed + notif client
      - preparer 123   -> preparing + notif client
      - pret 123       -> ready + notif client
      - livre 123      -> delivered + notif client
      - annule 123     -> cancelled + notif client
    Retourne un accusé au restaurant, ou None si pas de commande reconnue.
    """
    t = normalize(text)
    m = re.search(r"(ok|confirmer|preparer|pret|ready|livre|delivre|annule|cancel)\s*#?\s*(\d+)", t)
    if not m:
        return None

    cmd = m.group(1)
    oid = int(m.group(2))

    svc = OrderService(db)
    order = svc.get_order(oid)
    if not order:
        return f"❌ Commande #{oid} introuvable."

    # récup client
    customer = db.query(Customer).filter(Customer.id == order.customer_id).first()
    client_phone = customer.phone_number if customer else None

    # map status
    if cmd in ("ok", "confirmer"):
        new_status = OrderStatus.CONFIRMED
        client_msg = f"✅ Votre commande #{oid} est *confirmée* et passe en préparation."
    elif cmd in ("preparer",):
        new_status = OrderStatus.PREPARING
        client_msg = f"👨‍🍳 Votre commande #{oid} est *en préparation*."
    elif cmd in ("pret", "ready"):
        new_status = OrderStatus.READY
        client_msg = f"🎉 Votre commande #{oid} est *prête*. Vous pouvez venir la récupérer."
    elif cmd in ("livre", "delivre"):
        new_status = OrderStatus.DELIVERED
        client_msg = f"📦 Votre commande #{oid} a été *livrée*. Merci !"
    elif cmd in ("annule", "cancel"):
        new_status = OrderStatus.CANCELLED
        client_msg = f"❌ Votre commande #{oid} a été *annulée*. Contactez-nous pour plus d'infos."
    else:
        return None

    svc.set_status(order, new_status)

    # notifie le client si possible
    if client_phone:
        whatsapp.send_message(client_phone, client_msg)

    return f"✅ Statut commande #{oid} → {new_status}"

# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------
app = FastAPI(title="WhatsApp AI Agent - Système de Commandes")

@app.on_event("startup")
def _seed_on_startup():
    try:
        init_sample_data()
        logging.info("Startup seeding done.")
    except Exception as e:
        logging.error(f"Init sample data failed: {e}", exc_info=True)

@app.get("/")
async def root():
    return {"message": "WhatsApp AI Agent actif!", "status": "running"}

@app.get("/webhook")
async def verify_webhook(request: Request):
    verify_token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    if verify_token == config.WHATSAPP_VERIFY_TOKEN:
        return int(challenge)
    raise HTTPException(status_code=403, detail="Token invalide")

@app.post("/webhook")
async def handle_webhook(request: Request, db: Session = Depends(get_db)):
    try:
        body = await request.json()
        logging.info(f"INCOMING: {json.dumps(body)[:1200]}")

        entries = body.get("entry", [])
        if not entries:
            return JSONResponse({"status": "ignored"})

        wa = WhatsAppService()
        processed = False

        for entry in entries:
            for change in entry.get("changes", []):
                value = change.get("value", {})
                messages = value.get("messages", [])
                statuses = value.get("statuses", [])

                if statuses and not messages:
                    logging.debug(f"WA STATUS ONLY: {json.dumps(statuses)[:800]}")
                    continue

                for msg in messages:
                    from_number = msg.get("from")
                    mtype = msg.get("type")

                    # Si c'est le numéro du restaurant, traiter comme commande admin
                    if from_number == config.RESTAURANT_PHONE and mtype == "text":
                        text = (msg.get("text") or {}).get("body", "") or ""
                        ack = process_admin_command(db, text, wa)
                        if ack:
                            wa.send_message(config.RESTAURANT_PHONE, ack)
                            processed = True
                        continue

                    # Sinon, flux client normal
                    conv = ConversationService(db)

                    if mtype == "text":
                        text = (msg.get("text") or {}).get("body", "") or ""
                        if text.strip():
                            reply = conv.process_incoming_message(from_number, text.strip())
                            wa.send_message(from_number, reply)
                            processed = True

                    elif mtype == "interactive":
                        interactive = msg.get("interactive", {})
                        if "list_reply" in interactive:
                            lr = interactive["list_reply"]
                            lr_id = lr.get("id", "")
                            title = lr.get("title", "")
                            reply = conv.process_interactive_reply(from_number, lr_id, title)
                            wa.send_message(from_number, reply)
                            processed = True

        return JSONResponse({"status": "success" if processed else "ok-empty"})

    except Exception as e:
        logging.exception(f"Erreur webhook: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# -----------------------------------------------------------------------------
# Init + run
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

def init_sample_data():
    db = SessionLocal()
    try:
        if db.query(Product).count() == 0:
            products = [
                Product(name="Pizza Margherita", description="Tomate, mozzarella, basilic", price=12.0, category="Pizza",  available="true"),
                Product(name="Pizza Pepperoni",  description="Tomate, mozzarella, pepperoni", price=14.0, category="Pizza",  available="true"),
                Product(name="Pasta Carbonara",  description="Pâtes, lardons, crème, parmesan", price=10.0, category="Pasta", available="true"),
                Product(name="Salade César",     description="Salade, poulet, parmesan, croûtons", price=8.0,  category="Salade", available="true"),
                Product(name="Coca-Cola",        description="Boisson gazeuse 33cl", price=3.0,  category="Boisson", available="true"),
                Product(name="Eau minérale 50cl",description="Eau minérale 50cl", price=2.0,  category="Boisson", available="true"),
            ]
            for p in products:
                db.add(p)
            db.commit()
            logging.info("✅ Données de test initialisées.")
    finally:
        db.close()

if __name__ == "__main__":
    init_sample_data()
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
 