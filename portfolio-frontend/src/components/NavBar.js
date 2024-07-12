// src/components/NavBar.js
import React from 'react';
import { Navbar, Nav } from 'react-bootstrap';
import './NavBar.css'; // Custom CSS for the Navbar

const NavBar = () => {
    return (
        <Navbar bg="light" expand="lg" className="custom-navbar">
            <Navbar.Brand href="#home" className="custom-brand">
                Portfolio Optimization
            </Navbar.Brand>
            <Navbar.Toggle aria-controls="basic-navbar-nav" />
            <Navbar.Collapse id="basic-navbar-nav" className="justify-content-center">
                <Nav className="custom-nav">
                    <Nav.Link href="#home" className="custom-nav-link">Home</Nav.Link>
                    <Nav.Link href="#about" className="custom-nav-link">About</Nav.Link>
                </Nav>
            </Navbar.Collapse>
        </Navbar>
    );
};

export default NavBar;
