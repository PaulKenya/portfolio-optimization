// src/App.js
import React, { useState, useEffect } from 'react';
import NavBar from './components/NavBar';
import HomePage from './pages/HomePage';
import { getRequests, createRequest, deleteRequest as apiDeleteRequest } from './services/api';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css'; // Custom CSS for the whole app

const App = () => {
    const [requests, setRequests] = useState([]);

    useEffect(() => {
        const fetchData = async () => {
            const result = await getRequests();
            setRequests(result);
        };
        fetchData();
    }, []);

    const addRequest = async (request) => {
        const newRequest = await createRequest(request);
        setRequests([...requests, newRequest]);
    };

    const deleteRequest = async (index) => {
        const requests = await apiDeleteRequest(index);
        setRequests(requests);
    };

    return (
        <div>
            <NavBar />
            <HomePage addRequest={addRequest} requests={requests} deleteRequest={deleteRequest} />
        </div>
    );
};

export default App;
