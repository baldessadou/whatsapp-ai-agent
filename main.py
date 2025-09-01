# Agent IA WhatsApp - Système de Commandes
# Architecture complète avec FastAPI, SQLAlchemy, et intégration WhatsApp Business API

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

# Dépendances requises
"""
pip install fastapi uvicorn sqlalchemy psycopg2-binary
pip install requests python-dotenv openai anthropic
pip install python-multipart jinja2
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
import requests
from openai import OpenAI
import anthropic

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

# Service d'IA pour traitement du langage naturel
class AIService:
    def __init__(self):
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        
    def process_message(self, message: str, context: Dict = None) -> Dict:
        """Traite le message avec l'IA pour extraire l'intention et les entités"""
        
        system_prompt = """
        Tu es un assistant IA pour un système de commandes via WhatsApp.
        Analyse le message du client et réponds en JSON avec:
        - intent: "order", "inquiry", "modify_order", "cancel_order", "greeting", "other"
        - entities: objets extraits (produits, quantités, etc.)
        - response: réponse à envoyer au client
        - action_needed: action spécifique requise
        
        Produits disponibles:
        - Pizza Margherita (€12)
        - Pizza Pepperoni (€14)
        - Pasta Carbonara (€10)
        - Salade César (€8)
        - Coca-Cola (€3)
        - Eau (€2)
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Message: {message}\nContexte: {context}"}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logging.error(f"Erreur IA: {e}")
            return {
                "intent": "other",
                "entities": {},
                "response": "Désolé, je n'ai pas compris. Pouvez-vous reformuler?",
                "action_needed": "clarification"
            }

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
        self.ai_service = AIService()
        self.whatsapp_service = WhatsAppService()
        self.order_service = OrderService(db)
        
    def get_conversation_context(self, phone_number: str) -> Dict:
        """Récupère le contexte de conversation"""
        conv = self.db.query(Conversation).filter(Conversation.phone_number == phone_number).first()
        if conv and conv.context:
            return json.loads(conv.context)
        return {"state": "new", "current_order": [], "step": "greeting"}
    
    def update_conversation_context(self, phone_number: str, context: Dict):
        """Met à jour le contexte de conversation"""
        conv = self.db.query(Conversation).filter(Conversation.phone_number == phone_number).first()
        if not conv:
            conv = Conversation(phone_number=phone_number)
            self.db.add(conv)
        
        conv.context = json.dumps(context)
        conv.last_interaction = datetime.utcnow()
        self.db.commit()
    
    def process_incoming_message(self, phone_number: str, message: str) -> str:
        """Traite un message entrant et génère une réponse"""
        context = self.get_conversation_context(phone_number)
        
        # Analyser le message avec l'IA
        ai_response = self.ai_service.process_message(message, context)
        
        response = ""
        
        if ai_response["intent"] == "greeting":
            response = "🍕 Bonjour! Bienvenue chez Barita Resto. Tapez 'menu' pour voir nos plats ou décrivez ce que vous souhaitez commander!"
            context["state"] = "menu_requested"
            
        elif ai_response["intent"] == "inquiry" and "menu" in message.lower():
            # Envoyer le menu interactif
            products = self.db.query(Product).filter(Product.available == "true").all()
            products_dict = [{"id": p.id, "name": p.name, "description": p.description, "price": p.price} for p in products]
            
            ok = self.whatsapp_service.send_interactive_menu(phone_number, products_dict)
            if not ok:
                # fallback en texte simple
                lines = ["🍕 *Notre menu*"]
                for p in products[:10]:
                    lines.append(f"• {p.name} — €{p.price}")
                lines.append("\nRépondez par ex. : 2 margherita, 1 coca")
                self.whatsapp_service.send_message(phone_number, "\n".join(lines))
            
            response = "📋 Menu envoyé ! Vous pouvez aussi me dire directement ce que vous voulez."
            
        elif ai_response["intent"] == "order":
            # Traiter la commande
            entities = ai_response.get("entities", {})
            if "items" in entities:
                context["current_order"].extend(entities["items"])
                total = sum(item.get("price", 0) * item.get("quantity", 1) for item in context["current_order"])
                
                order_summary = "\n".join([f"• {item['quantity']}x {item['name']} - €{item['price']*item['quantity']}" 
                                         for item in context["current_order"]])
                
                response = f"✅ Ajouté à votre commande!\n\n📋 Récapitulatif:\n{order_summary}\n\n💰 Total: €{total:.2f}\n\nTapez 'confirmer' pour valider ou continuez à ajouter des articles."
                context["state"] = "order_building"
            else:
                response = "Je n'ai pas bien compris votre commande. Pouvez-vous préciser les articles et quantités? Par exemple: '2 pizzas margherita'"
                
        elif ai_response["intent"] == "modify_order" and "confirmer" in message.lower():
            if context["current_order"]:
                # Créer la commande
                order = self.order_service.create_order(phone_number, context["current_order"])
                response = f"🎉 Commande confirmée! Numéro: #{order.id}\n\n⏰ Temps de préparation: 25-30 minutes\n💰 Total: €{order.total_amount:.2f}\n\nVous recevrez une notification quand c'est prêt!"
                context = {"state": "order_confirmed", "current_order": [], "last_order_id": order.id}
            else:
                response = "Votre panier est vide. Ajoutez des articles avant de confirmer!"
                
        else:
            response = ai_response.get("response", "Je n'ai pas compris. Tapez 'menu' pour voir nos options!")
        
        # Sauvegarder le contexte
        self.update_conversation_context(phone_number, context)
        
        return response

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
        
        if "messages" in body.get("entry", [{}])[0].get("changes", [{}])[0].get("value", {}):
            messages = body["entry"][0]["changes"][0]["value"]["messages"]
            
            for message in messages:
                phone_number = message["from"]
                message_body = message.get("text", {}).get("body", "")
                
                if message_body:  # Ignorer les messages vides
                    conversation_service = ConversationService(db)
                    response = conversation_service.process_incoming_message(phone_number, message_body)
                    
                    # Envoyer la réponse
                    whatsapp_service = WhatsAppService()
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