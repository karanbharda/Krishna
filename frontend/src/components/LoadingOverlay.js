import React from 'react';
import styled, { keyframes } from 'styled-components';

const spin = keyframes`
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
`;

const Overlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
`;

const LoadingSpinner = styled.div`
  text-align: center;
  color: white;

  i {
    font-size: 3rem;
    margin-bottom: 15px;
    animation: ${spin} 1s linear infinite;
  }

  p {
    font-size: 1.2rem;
    margin: 0;
  }
`;

const LoadingOverlay = () => {
  return (
    <Overlay>
      <LoadingSpinner>
        <i className="fas fa-spinner"></i>
        <p>Processing...</p>
      </LoadingSpinner>
    </Overlay>
  );
};

export default LoadingOverlay;
