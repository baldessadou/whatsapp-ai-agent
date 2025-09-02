# main.py
# WhatsApp AI Agent – Commandes (R3)
# FastAPI + SQLAlchemy + WhatsApp Business Cloud API v22
# - Menu interactif + fallback texte
# - Parsing robuste ("2 margherita", "2x carbonara", "2 margherita et 1 coca")
# - Réponse aux list replies
# - Fallback produits si "available" manquant

import os
import re
import json
import logging
import unicodedata
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship

import requests

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class Config:
    WHATSAPP_TOKEN: str = os.getenv("WHATSAPP_TOKEN", "your_whatsapp_token")
    WHATSAPP_PHONE_ID: str = os.getenv("WHATSAPP_PHONE_ID", "your_phone_id")
    WHATSAPP_VERIFY_TOKEN: str = os.getenv("WHATSAPP_VERIFY_TOKEN", "verify_token_123")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./whatsapp_orders.db")

config = Config()

# -----------------------------------------------------------------------------
# DB
# -----------------------------------------------------------------------------
engine = create_engine(config.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class OrderStatus(Enum):
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
    # NOTE: string "true"/"false" pour rester compatible avec ta DB existante
    available = Column(String, default="true")

class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"))
    status = Column(String, default=OrderStatus.PENDING.value)
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
        # Construit les lignes; si pas de produits, on retourne False (fallback)
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
            logging.warning("No products to compose interactive menu; skip WA interactive.")
            return False

        url = f"{self.base_url}/messages"
        data = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "interactive",
            "interactive": {
                "type": "list",
                "header": {"type": "text", "text": "🍕 Menu Restaurant"},
                "body": {"text": "Choisissez vos articles :"},
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
        )
        self.db.add(order)
        self.db.commit()
        self.db.refresh(order)
        return order

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
        # Essaye d'abord available == "true"
        prods = self.db.query(Product).filter(Product.available == "true").all()
        if not prods:
            # Fallback : retourne tous les produits (DB ancienne sans "available")
            prods = self.db.query(Product).all()
        logging.info(f"Loaded products: {len(prods)} (available=='true' fallback if needed)")
        return prods

    # ---- synonymes (mapping texte -> produit)
    def _synonyms_map(self) -> Dict[str, Product]:
        m: Dict[str, Product] = {}
        for p in self._all_available_products():
            full = normalize(p.name)                 # "pizza margherita"
            words = [w for w in full.split() if w]
            if not words:
                continue
            last = words[-1]                         # "margherita"

            # entrées de base
            m[full] = p
            m[last] = p

            # variantes usuelles / raccourcis
            if "pepperoni" in full:
                m["pizza pepperoni"] = p
                m["pepperoni"] = p
            if "margherita" in full:
                m["pizza margherita"] = p
                m["margherita"] = p
            if "carbonara" in full:
                m["carbonara"] = p
                m["pasta carbonara"] = p
                m["pates carbonara"] = p
            if "cesar" in full or "cesar" in last:
                m["salade cesar"] = p
                m["cesar"] = p
            if "coca" in full:
                m["coca"] = p
                m["coca cola"] = p
            if "eau" in full:
                m["eau"] = p
                m["eau minerale"] = p

        logging.info(f"Synonyms keys: {len(m)}")
        return m

    # ---- intent
    def _detect_intent(self, msg: str) -> str:
        m = normalize(msg)
        if any(w in m for w in ("bonjour", "salut", "hello", "coucou")):
            return "greeting"
        if "menu" in m:
            return "menu"
        if "confirmer" in m or "valider" in m:
            return "confirm"
        if self._parse_items(m):
            return "order"
        return "other"

    # ---- parsing
    def _split_phrases(self, m: str) -> List[str]:
        # Sépare sur virgules, points-virgules, plus, et "et"
        m = re.sub(r"\s*(,|;|\+|\bet\b)\s*", "|", m)
        parts = [p.strip() for p in m.split("|") if p.strip()]
        return parts or [m]

    def _qty_in_text(self, s: str) -> int:
        # attrape "2", "2x", "2 x", "×2", ou "margherita 2"
        m = re.search(r"(\d+)\s*(?:x|×)?", s)
        if m:
            return max(1, int(m.group(1)))
        # fin de chaîne "margherita 2"
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
            # supprime quantité au début si présente
            chunk_wo_qty = re.sub(r"^\s*\d+\s*(?:x|×)?\s*", "", chunk).strip()

            picked: Optional[Product] = None
            for k in keys:
                if k and k in chunk_wo_qty:
                    picked = syn[k]
                    break

            if picked:
                items.append({
                    "name": picked.name,
                    "price": float(picked.price),
                    "quantity": qty
                })

        logging.info(f"Parsed items from '{msg_norm}': {items}")
        return items

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
            products_dict = [
                {"id": p.id, "name": p.name, "description": p.description, "price": p.price}
                for p in products
            ]
            ok = self.whatsapp.send_interactive_menu(phone, products_dict)
            if not ok:
                lines = ["🍕 *Notre menu*"]
                for p in products[:10]:
                    lines.append(f"• {p.name} — €{p.price:.2f}")
                lines.append("\nRépondez par ex. : 2 margherita, 1 coca")
                self.whatsapp.send_message(phone, "\n".join(lines))
            response = "📋 Menu envoyé ! Vous pouvez aussi me dire directement ce que vous voulez."
            context["state"] = "menu_shown"

        elif intent == "order":
            items = self._parse_items(normalize(message))
            if items:
                context.setdefault("current_order", [])
                context["current_order"].extend(items)
                total = sum(i["price"] * i["quantity"] for i in context["current_order"])
                lines = [
                    f"• {i['quantity']}× {i['name']} — €{i['price'] * i['quantity']:.2f}"
                    for i in context["current_order"]
                ]
                response = ("✅ Ajouté à votre commande !\n\n"
                            "📋 *Récapitulatif*:\n" + "\n".join(lines) +
                            f"\n\n💰 *Total*: €{total:.2f}\n"
                            "Tapez *confirmer* pour valider, ou continuez à ajouter des articles.")
                context["state"] = "order_building"
            else:
                response = ("Je n'ai pas bien compris les articles. Donnez un format comme : "
                            "*2 margherita et 1 coca*.")

        elif intent == "confirm":
            cart = context.get("current_order", [])
            if cart:
                order = self.order_service.create_order(phone, cart)
                response = (f"🎉 Commande confirmée ! Numéro: #{order.id}\n"
                            "⏰ Temps de préparation: 25–30 minutes\n"
                            f"💰 Total: €{order.total_amount:.2f}\n"
                            "Vous recevrez une notification quand c'est prêt !")
                context["state"] = "order_confirmed"
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
                context.setdefault("current_order", [])
                context["current_order"].append(item)

                total = sum(i["price"] * i["quantity"] for i in context["current_order"])
                lines = [
                    f"• {i['quantity']}× {i['name']} — €{i['price'] * i['quantity']:.2f}"
                    for i in context["current_order"]
                ]
                response = ("✅ Ajouté au panier depuis le menu.\n\n"
                            "📋 *Récapitulatif*:\n" + "\n".join(lines) +
                            f"\n\n💰 *Total*: €{total:.2f}\n"
                            "Tapez *confirmer* pour valider, ou continuez à ajouter des articles.")
                context["state"] = "order_building"
                self.update_conversation_context(phone, context)
                return response

        return ("Je n'ai pas pu ajouter cet élément. Réessayez depuis le *menu* "
                "ou envoyez un message du type *1 margherita*.")

# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------
app = FastAPI(title="WhatsApp AI Agent - Système de Commandes")

@app.on_event("startup")
def _seed_on_startup():
    try:
        init_sample_data()   # ne fait rien si les produits existent déjà
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

        conv = ConversationService(db)
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

                    if mtype == "text":
                        text = (msg.get("text") or {}).get("body", "")
                        text = (text or "").strip()
                        logging.info(f"INCOMING MESSAGE [text] from {from_number}: {text!r}")
                        if text:
                            reply = conv.process_incoming_message(from_number, text)
                            WhatsAppService().send_message(from_number, reply)
                            processed = True

                    elif mtype == "interactive":
                        interactive = msg.get("interactive", {})
                        if "list_reply" in interactive:
                            lr = interactive["list_reply"]
                            lr_id = lr.get("id", "")
                            title = lr.get("title", "")
                            logging.info(f"INCOMING MESSAGE [list_reply] {lr_id} / {title}")
                            reply = conv.process_interactive_reply(from_number, lr_id, title)
                            WhatsAppService().send_message(from_number, reply)
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
        else:
            # Optionnel : log si les produits existants n'ont pas 'available' test 
            missing = db.query(Product).filter((Product.available.is_(None)) | (Product.available == "")).count()
            if missing:
                logging.warning(f"{missing} produits sans 'available' défini (fallback all-products activé).")
    finally: 
        db.close()

if __name__ == "__main__":
    init_sample_data()
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
 