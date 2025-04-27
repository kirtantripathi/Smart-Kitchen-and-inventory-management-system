import React, { useState } from 'react';
import axios from 'axios';
import { marked } from 'marked';

const ChefChat = () => {
  const [input, setInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [sessionId, setSessionId] = useState(null);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { sender: 'user', message: input };
    setChatHistory(prev => [...prev, userMessage]);

    try {
      const res = await axios.post('http://localhost:5000/chat', {
        message: input,
        session_id: sessionId
      });

      const botMessage = { sender: 'chef', message: res.data.response };
      setChatHistory(prev => [...prev, botMessage]);
      setSessionId(res.data.session_id);
      setInput('');
    } catch (err) {
      console.error('Error communicating with chatbot:', err);
    }
  };

  const handleKeyDown = e => {
    if (e.key === 'Enter') sendMessage();
  };

  return (
    <div style={styles.chatContainer}>
      <h2 style={styles.header}>👨‍🍳 Chef Bot</h2>
      <div style={styles.chatBox}>
        {chatHistory.map((msg, idx) => (
          <div key={idx} style={msg.sender === 'user' ? styles.userMessage : styles.chefMessage}>
            <strong style={styles.messageSender}>{msg.sender === 'user' ? 'You' : 'Chef'}:</strong>
            <div
              style={styles.messageContent}
              dangerouslySetInnerHTML={{ __html: marked.parse(msg.message) }}
            />
          </div>
        ))}
      </div>
      <div style={styles.inputBox}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a cooking question..."
          style={styles.inputField}
        />
        <button onClick={sendMessage} style={styles.sendButton}>Send</button>
      </div>
    </div>
  );
};

const styles = {
  chatContainer: {
    width: '100%',
    maxWidth: '1200px',
    margin: '0 auto',
    borderRadius: '10px',
    backgroundColor: '#f9f9f9',
    padding: '20px',
    boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
  },
  header: {
    textAlign: 'center',
    color: '#333',
    marginBottom: '20px',
    fontFamily: 'Arial, sans-serif',
  },
  chatBox: {
    maxHeight: '400px',
    overflowY: 'auto',
    marginBottom: '20px',
    paddingRight: '10px',
  },
  userMessage: {
    backgroundColor: '#d1f7c4',
    margin: '10px 0',
    padding: '10px',
    borderRadius: '10px',
    alignSelf: 'flex-end',
    maxWidth: '80%',
    width: 'auto', // or use a fixed width like '60%'
    textAlign: 'right',
    wordBreak: 'break-word',
    marginLeft: 'auto',
  },
  
  chefMessage: {
    backgroundColor: '#e1e1e1',
    margin: '10px 0',
    padding: '10px',
    borderRadius: '10px',
    alignSelf: 'flex-start',
    maxWidth: '80%',
    width: 'auto', // or '60%' for fixed width
    textAlign: 'left',
    wordBreak: 'break-word',
    marginRight: 'auto',
  },
  
  messageSender: {
    fontWeight: 'bold',
    color: '#444',
  },
  messageContent: {
    marginTop: '5px',
    fontSize: '14px',
    color: '#555',
  },
  inputBox: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: '10px',
  },
  inputField: {
    width: '80%',
    padding: '10px',
    fontSize: '16px',
    borderRadius: '20px',
    border: '1px solid #ccc',
    marginRight: '10px',
  },
  sendButton: {
    backgroundColor: '#4CAF50',
    color: 'white',
    border: 'none',
    padding: '12px 20px',
    fontSize: '16px',
    borderRadius: '20px',
    cursor: 'pointer',
    transition: 'background-color 0.3s',
  },
};

export default ChefChat;