#!/bin/bash

echo "📊 MONITOR RAG CHATBOT"
echo "======================"

# Estado de servicios
echo "🔧 Servicios:"

# Milvus
if curl -s http://localhost:19530/health > /dev/null; then
    echo "✅ Milvus: ACTIVO"
else
    echo "❌ Milvus: INACTIVO"
fi

# PostgreSQL
if docker ps | grep rag_postgres > /dev/null; then
    echo "✅ PostgreSQL: ACTIVO (Docker)"
elif pgrep -x postgres > /dev/null; then
    echo "✅ PostgreSQL: ACTIVO (Nativo)"
elif nc -z localhost 5432 2>/dev/null; then
    echo "✅ PostgreSQL: ACTIVO"
else
    echo "❌ PostgreSQL: INACTIVO"
fi

# Uso de memoria
echo ""
echo "💾 Memoria:"
free -h | awk 'NR==2{printf "   Usada: %s/%s (%.0f%%)\n", $3,$2,$3*100/$2}'

# Puertos
echo ""
echo "🌐 Puertos:"
if netstat -tlpn 2>/dev/null | grep ":19530 " > /dev/null; then
    echo "✅ 19530 (Milvus)"
else
    echo "❌ 19530 (Milvus)"
fi

if netstat -tlpn 2>/dev/null | grep ":5432 " > /dev/null || ss -tlpn 2>/dev/null | grep ":5432 " > /dev/null; then
    echo "✅ 5432 (PostgreSQL)"
else
    echo "❌ 5432 (PostgreSQL)"
fi