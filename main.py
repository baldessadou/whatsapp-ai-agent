# Agent IA WhatsApp - Système de Commandes (Version Corrigée)
# Architecture complète avec FastAPI, SQLAlchemy, et intégration WhatsApp Business API

import os
import re
import json
import logging
import unicodedata
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

# Dépendances requises
"""
pip install fastapi uvicorn sqlalchemy psycopg2-binary
pip install requests python-dotenv
pip install python-multipart
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
import requests

# Configuration
@dataclass
class Config:
    WHATSAPP_TOKEN: str = os.getenv("WHATSAPP_TOKEN", "your_whatsapp_token")
    WHATSAPP_PHONE_ID: str = os.getenv("WHATSAPP_PHONE_ID", "your_phone_id")
    WHATSAPP_VERIFY_TOKEN: str = os.getenv("WHATSAPP_VERIFY_TOKEN", "verify_token_123")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "your_openai_key")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./whatsapp_orders.db")
    
config = Config()

# Base de données
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

# Modèles de base de données
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
    available = Column(String, default="true")
    
class Order(Base):
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"))
    status = Column(String, default=OrderStatus.PENDING.value)
    total_amount = Column(Float)
    items = Column(Text)  # JSON string des articles
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    customer = relationship("Customer", back_populates="orders")

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String, index=True)
    context = Column(Text)  # Contexte de la conversation en JSON
    last_interaction = Column(DateTime, default=datetime.utcnow)

# Créer les tables
Base.metadata.create_all(bind=engine)

# Dépendance pour obtenir la session DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Utils pour parsing
def normalize(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.replace("-", " ").strip()

# Service de parsing simple (sans IA)
class OrderParser:
    def __init__(self, db: Session):
        self.db = db
        
    def get_product_synonyms(self) -> Dict[str, Product]:
        """Crée un mapping {synonyme_normalise -> produit}"""
        synonyms = {}
        products = self.db.query(Product).filter(Product.available == "true").all()
        
        for p in products:
            full_name = normalize(p.name)
            words = [w for w in full_name.split() if w]
            
            # Nom complet
            synonyms[full_name] = p
            
            # Dernier mot (ex: "margherita" pour "Pizza Margherita")
            if words:
                synonyms[words[-1]] = p
            
            # Synonymes spéciaux
            if "margherita" in full_name:
                synonyms["margherita"] = p
                synonyms["pizza margherita"] = p
            if "pepperoni" in full_name:
                synonyms["pepperoni"] = p
                synonyms["pizza pepperoni"] = p
            if "carbonara" in full_name:
                synonyms["carbonara"] = p
                synonyms["pasta carbonara"] = p
                synonyms["pates carbonara"] = p
            if "cesar" in full_name:
                synonyms["cesar"] = p
                synonyms["salade cesar"] = p
            if "coca" in full_name:
                synonyms["coca"] = p
                synonyms["cola"] = p
            if "eau" in full_name:
                synonyms["eau"] = p
                
        return synonyms
    
    def extract_quantity(self, text: str) -> int:
        """Extrait la quantité du texte (1 par défaut)"""
        match = re.search(r'(\d+)\s*(?:x|×)?', normalize(text))
        return max(1, int(match.group(1))) if match else 1
    
    def parse_order(self, message: str) -> List[Dict]:
        """Parse un message pour extraire les articles commandés"""
        message_norm = normalize(message)
        synonyms = self.get_product_synonyms()
        
        # Divise sur les séparateurs communs
        chunks = re.split(r'\s*(,|;|\+|\bet\b)\s*', message_norm)
        chunks = [c.strip() for c in chunks if c.strip() and c not in [',', ';', '+', 'et']]
        
        items = []
        # Trie les synonymes par longueur (plus long en premier)
        sorted_synonyms = sorted(synonyms.keys(), key=len, reverse=True)
        
        for chunk in chunks:
            quantity = self.extract_quantity(chunk)
            
            # Trouve le produit correspondant
            found_product = None
            for syn in sorted_synonyms:
                if syn in chunk:
                    found_product = synonyms[syn]
                    break
            
            if found_product:
                items.append({
                    "name": found_product.name,
                    "price": float(found_product.price),
                    "quantity": quantity
                })
        
        return items

# Service WhatsApp
class WhatsAppService:
    def __init__(self):
        self.token = config.WHATSAPP_TOKEN
        self.phone_id = config.WHATSAPP_PHONE_ID
        self.base_url = f"https://graph.facebook.com/v18.0/{self.phone_id}"
        
    def send_message(self, to: str, message: str) -> bool:
        """Envoie un message texte"""
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": message}
        }
        
        try:
            logging.info(f"WA token head: {self.token[:6]}..., phone_id={self.phone_id}")
            response = requests.post(url, json=data, headers=headers, timeout=15)
            if response.status_code in (200, 201):
                logging.info(f"WA send ok: {response.text}")
                return True
            else:
                logging.error(f"WA send failed {response.status_code}: {response.text}")
                return False
        except Exception as e:
            logging.error(f"Erreur envoi message: {e}")
            return False
    
    def send_interactive_menu(self, to: str, products: List[Dict]) -> bool:
        """Envoie un menu interactif"""
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        # Préparer les lignes du menu
        rows = []
        for p in products[:10]:
            title = p.get("name", "Article")
            desc = f"{p.get('description', '')}".strip()
            price = p.get("price")
            if price is not None:
                desc = (desc + (" - " if desc else "")) + f"€{price}"
            rows.append({
                "id": f"product_{p.get('id', title)}",
                "title": title[:24],
                "description": desc[:72]
            })
        
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
            response = requests.post(url, json=data, headers=headers, timeout=15)
            ok = response.status_code in (200, 201)
            if ok:
                logging.info(f"WA interactive ok: {response.text}")
            else:
                logging.error(f"WA interactive failed {response.status_code}: {response.text}")
            return ok
        except Exception as e:
            logging.error(f"Erreur menu interactif: {e}")
            return False

# Service de gestion des commandes
class OrderService:
    def __init__(self, db: Session):
        self.db = db
        
    def get_or_create_customer(self, phone_number: str) -> Customer:
        """Récupère ou crée un client"""
        customer = self.db.query(Customer).filter(Customer.phone_number == phone_number).first()
        if not customer:
            customer = Customer(phone_number=phone_number)
            self.db.add(customer)
            self.db.commit()
            self.db.refresh(customer)
        return customer
    
    def create_order(self, phone_number: str, items: List[Dict], notes: str = "") -> Order:
        """Crée une nouvelle commande"""
        customer = self.get_or_create_customer(phone_number)
        
        # Calculer le total
        total = sum(item['price'] * item['quantity'] for item in items)
        
        order = Order(
            customer_id=customer.id,
            total_amount=total,
            items=json.dumps(items),
            notes=notes
        )
        
        self.db.add(order)
        self.db.commit()
        self.db.refresh(order)
        return order
    
    def get_customer_orders(self, phone_number: str) -> List[Order]:
        """Récupère les commandes d'un client"""
        customer = self.get_or_create_customer(phone_number)
        return self.db.query(Order).filter(Order.customer_id == customer.id).all()

# Service principal de conversation
class ConversationService:
    def __init__(self, db: Session):
        self.db = db
        self.whatsapp_service = WhatsAppService()
        self.order_service = OrderService(db)
        self.parser = OrderParser(db)
        
    def get_conversation_context(self, phone_number: str) -> Dict:
        """Récupère le contexte de conversation"""
        conv = self.db.query(Conversation).filter(Conversation.phone_number == phone_number).first()
        if conv and conv.context:
            return json.loads(conv.context)
        return {"state": "new", "current_order": []}
    
    def update_conversation_context(self, phone_number: str, context: Dict):
        """Met à jour le contexte de conversation"""
        conv = self.db.query(Conversation).filter(Conversation.phone_number == phone_number).first()
        if not conv:
            conv = Conversation(phone_number=phone_number)
            self.db.add(conv)
        
        conv.context = json.dumps(context)
        conv.last_interaction = datetime.utcnow()
        self.db.commit()
    
    def detect_intent(self, message: str) -> str:
        """Détecte l'intention du message"""
        msg_norm = normalize(message)
        
        # Salutations
        if any(word in msg_norm for word in ["bonjour", "salut", "hello", "coucou", "bonsoir"]):
            return "greeting"
        
        # Menu
        if "menu" in msg_norm:
            return "menu"
        
        # Confirmation
        if any(word in msg_norm for word in ["confirmer", "valider", "oui", "ok", "commander"]):
            return "confirm"
        
        # Commande (si on trouve des produits)
        items = self.parser.parse_order(message)
        if items:
            return "order"
        
        return "other"
    
    def process_incoming_message(self, phone_number: str, message: str) -> str:
        """Traite un message entrant et génère une réponse"""
        context = self.get_conversation_context(phone_number)
        intent = self.detect_intent(message)
        
        logging.info(f"[{phone_number}] Intent: {intent}, Message: {message}")
        
        if intent == "greeting":
            response = ("🍕 Bonjour! Bienvenue chez Barita Resto.\n\n"
                       "Tapez *menu* pour voir notre carte, ou dites-moi directement votre commande !\n"
                       "Exemple: *2 margherita et 1 coca*")
            context["state"] = "greeted"
            
        elif intent == "menu":
            products = self.db.query(Product).filter(Product.available == "true").all()
            products_dict = [
                {"id": p.id, "name": p.name, "description": p.description, "price": p.price} 
                for p in products
            ]
            
            # Tenter d'envoyer le menu interactif
            success = self.whatsapp_service.send_interactive_menu(phone_number, products_dict)
            
            if success:
                response = ("📋 Voici notre menu interactif ! Cliquez sur \"Voir menu\" pour choisir.\n\n"
                           "Vous pouvez aussi me dire directement: *2 margherita et 1 coca*")
            else:
                # Menu texte de fallback
                lines = ["🍕 **Notre Menu**"]
                for p in products:
                    lines.append(f"• {p.name} - €{p.price}")
                lines.append("\nPour commander, tapez par exemple: *2 margherita et 1 coca*")
                response = "\n".join(lines)
            
            context["state"] = "menu_shown"
            
        elif intent == "order":
            items = self.parser.parse_order(message)
            if items:
                context.setdefault("current_order", [])
                context["current_order"].extend(items)
                
                # Calculer le total
                total = sum(item["price"] * item["quantity"] for item in context["current_order"])
                
                # Créer le récapitulatif
                order_lines = []
                for item in context["current_order"]:
                    subtotal = item["price"] * item["quantity"]
                    order_lines.append(f"• {item['quantity']}x {item['name']} - €{subtotal:.2f}")
                
                response = ("✅ Ajouté à votre commande !\n\n"
                           "📋 **Récapitulatif:**\n" + "\n".join(order_lines) + 
                           f"\n\n💰 **Total: €{total:.2f}**\n\n"
                           "Tapez *confirmer* pour valider votre commande, "
                           "ou continuez à ajouter des articles.")
                
                context["state"] = "order_building"
            else:
                response = ("Je n'ai pas reconnu les articles demandés. 🤔\n\n"
                           "Essayez avec le format: *2 margherita et 1 coca*\n"
                           "Ou tapez *menu* pour voir tous nos produits.")
                
        elif intent == "confirm":
            current_order = context.get("current_order", [])
            if current_order:
                # Créer la commande en base
                order = self.order_service.create_order(phone_number, current_order)
                
                response = ("🎉 **Commande confirmée !**\n\n"
                           f"📋 Numéro de commande: **#{order.id}**\n"
                           f"💰 Total: **€{order.total_amount:.2f}**\n"
                           "⏰ Temps de préparation: **25-30 minutes**\n\n"
                           "Vous recevrez une notification quand votre commande sera prête ! 🍕")
                
                # Reset du contexte
                context["state"] = "order_confirmed"
                context["current_order"] = []
                context["last_order_id"] = order.id
            else:
                response = ("Votre panier est vide ! 🛒\n\n"
                           "Tapez *menu* pour voir nos produits ou dites-moi ce que vous voulez commander.")
                
        else:
            response = ("Je n'ai pas compris votre message. 😅\n\n"
                       "Tapez *menu* pour voir notre carte\n"
                       "ou essayez: *2 margherita et 1 coca*")
        
        # Sauvegarder le contexte
        self.update_conversation_context(phone_number, context)
        return response
    
    def process_interactive_reply(self, phone_number: str, list_reply_id: str, title: str) -> str:
        """Traite une réponse de menu interactif"""
        context = self.get_conversation_context(phone_number)
        
        # Extraire l'ID du produit
        if list_reply_id.startswith("product_"):
            try:
                product_id = int(list_reply_id.split("_", 1)[1])
                product = self.db.query(Product).filter(Product.id == product_id).first()
                
                if product:
                    item = {
                        "name": product.name,
                        "price": float(product.price),
                        "quantity": 1
                    }
                    
                    context.setdefault("current_order", [])
                    context["current_order"].append(item)
                    
                    # Calculer le total
                    total = sum(i["price"] * i["quantity"] for i in context["current_order"])
                    
                    # Récapitulatif
                    order_lines = []
                    for i in context["current_order"]:
                        subtotal = i["price"] * i["quantity"]
                        order_lines.append(f"• {i['quantity']}x {i['name']} - €{subtotal:.2f}")
                    
                    response = ("✅ **{} ajouté** à votre commande !\n\n"
                               "📋 **Récapitulatif:**\n" + "\n".join(order_lines) +
                               f"\n\n💰 **Total: €{total:.2f}**\n\n"
                               "Tapez *confirmer* pour valider ou continuez à ajouter des articles.").format(product.name)
                    
                    context["state"] = "order_building"
                    self.update_conversation_context(phone_number, context)
                    return response
                    
            except Exception as e:
                logging.error(f"Erreur traitement interactive reply: {e}")
        
        return "Problème avec la sélection. Réessayez depuis le menu ou tapez votre commande directement."

# Application FastAPI
app = FastAPI(title="WhatsApp AI Agent - Système de Commandes")

# Endpoints
@app.get("/")
async def root():
    return {"message": "WhatsApp AI Agent actif!", "status": "running"}

@app.get("/webhook")
async def verify_webhook(request: Request):
    """Vérification du webhook WhatsApp"""
    verify_token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    
    if verify_token == config.WHATSAPP_VERIFY_TOKEN:
        return int(challenge)
    else:
        raise HTTPException(status_code=403, detail="Token invalide")

@app.post("/webhook")
async def handle_webhook(request: Request, db: Session = Depends(get_db)):
    """Traite les messages WhatsApp entrants"""
    try:
        body = await request.json()
        logging.info(f"WEBHOOK REÇU: {json.dumps(body)[:1000]}")
        
        entries = body.get("entry", [])
        if not entries:
            return JSONResponse(content={"status": "no_entries"})
        
        conv = ConversationService(db)
        whatsapp_service = WhatsAppService()
        
        for entry in entries:
            changes = entry.get("changes", [])
            for change in changes:
                value = change.get("value", {})
                messages = value.get("messages", [])
                
                # Ignorer les statuts de lecture/livraison
                if not messages and value.get("statuses"):
                    continue
                
                for message in messages:
                    phone_number = message.get("from")
                    message_type = message.get("type")
                    
                    if message_type == "text":
                        # Message texte normal
                        message_body = message.get("text", {}).get("body", "")
                        if message_body.strip():
                            try:
                                response = conv.process_incoming_message(phone_number, message_body)
                                whatsapp_service.send_message(phone_number, response)
                            except Exception as e:
                                logging.error(f"Erreur traitement message: {e}")
                                whatsapp_service.send_message(phone_number, "Oups, petit problème. Réessayez ou tapez 'menu'.")
                    
                    elif message_type == "interactive":
                        # Réponse interactive (menu)
                        interactive = message.get("interactive", {})
                        
                        if "list_reply" in interactive:
                            # Réponse à un menu de liste
                            list_reply = interactive["list_reply"]
                            list_reply_id = list_reply.get("id", "")
                            title = list_reply.get("title", "")
                            
                            try:
                                response = conv.process_interactive_reply(phone_number, list_reply_id, title)
                                whatsapp_service.send_message(phone_number, response)
                            except Exception as e:
                                logging.error(f"Erreur traitement interactive: {e}")
                                whatsapp_service.send_message(phone_number, "Problème avec votre sélection. Réessayez depuis le menu.")
                        
                        elif "button_reply" in interactive:
                            # Réponse à un bouton (si vous en ajoutez)
                            button_reply = interactive["button_reply"]
                            button_id = button_reply.get("id", "")
                            title = button_reply.get("title", "")
                            
                            # Traiter comme un message texte
                            response = conv.process_incoming_message(phone_number, title or button_id)
                            whatsapp_service.send_message(phone_number, response)
        
        return JSONResponse(content={"status": "success"})
        
    except Exception as e:
        logging.error(f"Erreur webhook: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/admin/products")
async def create_product(name: str, description: str, price: float, category: str, db: Session = Depends(get_db)):
    """Créer un nouveau produit"""
    product = Product(name=name, description=description, price=price, category=category)
    db.add(product)
    db.commit()
    db.refresh(product)
    return {"message": "Produit créé", "product_id": product.id}

@app.get("/admin/orders")
async def get_orders(db: Session = Depends(get_db)):
    """Récupérer toutes les commandes"""
    orders = db.query(Order).all()
    return [{"id": o.id, "customer_phone": o.customer.phone_number, 
             "total": o.total_amount, "status": o.status, "created_at": o.created_at} for o in orders]

@app.put("/admin/orders/{order_id}/status")
async def update_order_status(order_id: int, status: str, db: Session = Depends(get_db)):
    """Mettre à jour le statut d'une commande"""
    order = db.query(Order).filter(Order.id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Commande introuvable")
    
    order.status = status
    order.updated_at = datetime.utcnow()
    db.commit()
    
    # Notifier le client
    whatsapp_service = WhatsAppService()
    status_messages = {
        "confirmed": "✅ Votre commande a été confirmée!",
        "preparing": "👨‍🍳 Votre commande est en préparation...",
        "ready": "🎉 Votre commande est prête! Vous pouvez venir la récupérer.",
        "delivered": "📦 Commande livrée! Merci et à bientôt!",
        "cancelled": "❌ Votre commande a été annulée. Contactez-nous pour plus d'infos."
    }
    
    if status in status_messages:
        whatsapp_service.send_message(
            order.customer.phone_number, 
            f"Commande #{order.id}: {status_messages[status]}"
        )
    
    return {"message": "Statut mis à jour"}

# Configuration de logging
logging.basicConfig(level=logging.INFO)

# Données de test pour initialiser la base
def init_sample_data():
    db = SessionLocal()
    try:
        if db.query(Product).count() == 0:
            products = [
                Product(name="Pizza Margherita", description="Tomate, mozzarella, basilic", price=12.0, category="Pizza"),
                Product(name="Pizza Pepperoni", description="Tomate, mozzarella, pepperoni", price=14.0, category="Pizza"),
                Product(name="Pasta Carbonara", description="Pâtes, lardons, crème, parmesan", price=10.0, category="Pasta"),
                Product(name="Salade César", description="Salade, poulet, parmesan, croûtons", price=8.0, category="Salade"),
                Product(name="Coca-Cola", description="Boisson gazeuse 33cl", price=3.0, category="Boisson"),
                Product(name="Eau", description="Eau minérale 50cl", price=2.0, category="Boisson"),
            ]
            
            for product in products:
                db.add(product)
            db.commit()
            print("✅ Données de test initialisées!")
    finally:
        db.close()

if __name__ == "__main__":
    init_sample_data()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)