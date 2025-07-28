import React, { useState, useEffect } from 'react';
import styled from 'styled-components';

const ModalOverlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
`;

const ModalContent = styled.div`
  background: white;
  border-radius: 12px;
  width: 90%;
  max-width: 500px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
  max-height: 90vh;
  overflow-y: auto;
`;

const ModalHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px;
  border-bottom: 1px solid #e9ecef;

  h3 {
    color: #2c3e50;
    margin: 0;
  }
`;

const CloseButton = styled.button`
  background: none;
  border: none;
  font-size: 1.2rem;
  cursor: pointer;
  color: #7f8c8d;
  padding: 5px;
  border-radius: 50%;
  width: 30px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;

  &:hover {
    color: #e74c3c;
    background: #f8f9fa;
  }
`;

const ModalBody = styled.div`
  padding: 20px;
`;

const SettingGroup = styled.div`
  margin-bottom: 20px;

  label {
    display: block;
    margin-bottom: 5px;
    font-weight: 500;
    color: #2c3e50;
  }

  select, input {
    width: 100%;
    padding: 10px;
    border: 2px solid #e9ecef;
    border-radius: 6px;
    font-size: 1rem;
    box-sizing: border-box;

    &:focus {
      outline: none;
      border-color: #3498db;
    }
  }

  input[type="number"] {
    -moz-appearance: textfield;
    
    &::-webkit-outer-spin-button,
    &::-webkit-inner-spin-button {
      -webkit-appearance: none;
      margin: 0;
    }
  }
`;

const ModalFooter = styled.div`
  padding: 20px;
  border-top: 1px solid #e9ecef;
  text-align: right;
  display: flex;
  gap: 10px;
  justify-content: flex-end;
`;

const Button = styled.button`
  padding: 10px 20px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 1rem;
  transition: all 0.3s ease;
  border: none;

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const SaveButton = styled(Button)`
  background: #27ae60;
  color: white;

  &:hover:not(:disabled) {
    background: #229954;
  }
`;

const CancelButton = styled(Button)`
  background: #95a5a6;
  color: white;

  &:hover:not(:disabled) {
    background: #7f8c8d;
  }
`;

const SettingsModal = ({ settings, onSave, onClose }) => {
  const [formData, setFormData] = useState({
    mode: 'paper',
    riskLevel: 'MEDIUM',
    maxAllocation: 25,
    stopLossPct: 5
  });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (settings) {
      setFormData({
        mode: settings.mode || 'paper',
        riskLevel: settings.riskLevel || 'MEDIUM',
        // Convert decimal to percentage - backend sends max_capital_per_trade and stop_loss_pct
        maxAllocation: settings.max_capital_per_trade
          ? (settings.max_capital_per_trade * 100)
          : 25,
        stopLossPct: settings.stop_loss_pct
          ? (settings.stop_loss_pct * 100)
          : 5
      });
    }
  }, [settings]);

  const handleInputChange = (field, value) => {
    setFormData(prev => {
      const newData = {
        ...prev,
        [field]: value
      };

      // Auto-update stop loss and allocation based on risk level
      if (field === 'riskLevel') {
        if (value === 'CUSTOM') {
          // When Custom is selected, clear the fields so user can input their own values
          newData.stopLossPct = '';
          newData.maxAllocation = '';
        } else {
          // For predefined risk levels, set the values
          const riskSettings = {
            'LOW': { stopLoss: 3, allocation: 15 },
            'MEDIUM': { stopLoss: 5, allocation: 25 },
            'HIGH': { stopLoss: 8, allocation: 35 }
          };

          if (riskSettings[value]) {
            newData.stopLossPct = riskSettings[value].stopLoss;
            newData.maxAllocation = riskSettings[value].allocation;
          }
        }
      }

      console.log('Risk Level:', newData.riskLevel, 'Is Custom:', newData.riskLevel === 'CUSTOM');
      return newData;
    });
  };

  const handleSave = async () => {
    setLoading(true);
    try {
      // Convert string values to numbers
      const maxAllocationNum = parseFloat(formData.maxAllocation) || 0;
      const stopLossPctNum = parseFloat(formData.stopLossPct) || 0;

      // Validate that we have values for custom mode
      if (formData.riskLevel === 'CUSTOM') {
        if (!formData.maxAllocation || !formData.stopLossPct || maxAllocationNum <= 0 || stopLossPctNum <= 0) {
          alert('Please enter valid values for both Max Allocation (1-100) and Stop Loss Percentage (1-20) when using Custom risk level.');
          setLoading(false);
          return;
        }

        // Validate ranges
        if (maxAllocationNum < 1 || maxAllocationNum > 100) {
          alert('Max Allocation must be between 1 and 100.');
          setLoading(false);
          return;
        }

        if (stopLossPctNum < 1 || stopLossPctNum > 20) {
          alert('Stop Loss Percentage must be between 1 and 20.');
          setLoading(false);
          return;
        }
      }

      // Use the converted numbers or defaults
      const maxAllocation = maxAllocationNum || 25;
      const stopLossPct = stopLossPctNum || 5;

      const settingsToSave = {
        mode: formData.mode,
        riskLevel: formData.riskLevel,
        stop_loss_pct: stopLossPct / 100, // Convert percentage to decimal
        max_capital_per_trade: maxAllocation / 100, // Convert percentage to decimal
        max_trade_limit: 10 // Default value
      };

      console.log('Saving settings:', settingsToSave);
      await onSave(settingsToSave);
    } catch (error) {
      console.error('Error saving settings:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleOverlayClick = (e) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <ModalOverlay onClick={handleOverlayClick}>
      <ModalContent>
        <ModalHeader>
          <h3>Settings</h3>
          <CloseButton onClick={onClose}>
            <i className="fas fa-times"></i>
          </CloseButton>
        </ModalHeader>

        <ModalBody>
          <SettingGroup>
            <label>Trading Mode:</label>
            <select
              value={formData.mode}
              onChange={(e) => handleInputChange('mode', e.target.value)}
              disabled={loading}
            >
              <option value="paper">Paper Trading</option>
              <option value="live">Live Trading</option>
            </select>
          </SettingGroup>

          <SettingGroup>
            <label>Risk Level:</label>
            <select
              value={formData.riskLevel}
              onChange={(e) => handleInputChange('riskLevel', e.target.value)}
              disabled={loading}
            >
              <option value="LOW">Low (3% stop-loss, 15% allocation)</option>
              <option value="MEDIUM">Medium (5% stop-loss, 25% allocation)</option>
              <option value="HIGH">High (8% stop-loss, 35% allocation)</option>
              <option value="CUSTOM">Custom (Set your own values)</option>
            </select>
          </SettingGroup>

          <SettingGroup>
            <label>Max Allocation per Trade (%):</label>
            <input
              type="number"
              min="1"
              max="100"
              value={formData.maxAllocation}
              placeholder={formData.riskLevel === 'CUSTOM' ? 'Enter percentage (1-100)' : ''}
              onChange={(e) => {
                const value = e.target.value;
                console.log('Max Allocation input changed:', value, 'Risk Level:', formData.riskLevel);
                // Always allow the change, let the input handle validation
                handleInputChange('maxAllocation', value);
              }}
              disabled={loading || formData.riskLevel !== 'CUSTOM'}
              style={{
                backgroundColor: formData.riskLevel !== 'CUSTOM' ? '#f8f9fa' : 'white',
                cursor: formData.riskLevel !== 'CUSTOM' ? 'not-allowed' : 'text',
                border: formData.riskLevel === 'CUSTOM' ? '2px solid #3498db' : '2px solid #e9ecef'
              }}
            />
            {formData.riskLevel !== 'CUSTOM' && (
              <small style={{ color: '#6c757d', fontSize: '0.85rem', marginTop: '5px', display: 'block' }}>
                Select "Custom" risk level to modify this value
              </small>
            )}
            {formData.riskLevel === 'CUSTOM' && (
              <small style={{ color: '#27ae60', fontSize: '0.85rem', marginTop: '5px', display: 'block' }}>
                ✓ Custom mode: You can edit this value
              </small>
            )}
          </SettingGroup>

          <SettingGroup>
            <label>Stop Loss Percentage (%):</label>
            <input
              type="number"
              min="1"
              max="20"
              step="0.1"
              value={formData.stopLossPct}
              placeholder={formData.riskLevel === 'CUSTOM' ? 'Enter percentage (1-20)' : ''}
              onChange={(e) => {
                const value = e.target.value;
                console.log('Stop Loss input changed:', value, 'Risk Level:', formData.riskLevel);
                // Always allow the change, let the input handle validation
                handleInputChange('stopLossPct', value);
              }}
              disabled={loading || formData.riskLevel !== 'CUSTOM'}
              style={{
                backgroundColor: formData.riskLevel !== 'CUSTOM' ? '#f8f9fa' : 'white',
                cursor: formData.riskLevel !== 'CUSTOM' ? 'not-allowed' : 'text',
                border: formData.riskLevel === 'CUSTOM' ? '2px solid #3498db' : '2px solid #e9ecef'
              }}
            />
            {formData.riskLevel !== 'CUSTOM' && (
              <small style={{ color: '#6c757d', fontSize: '0.85rem', marginTop: '5px', display: 'block' }}>
                Select "Custom" risk level to modify this value
              </small>
            )}
            {formData.riskLevel === 'CUSTOM' && (
              <small style={{ color: '#27ae60', fontSize: '0.85rem', marginTop: '5px', display: 'block' }}>
                ✓ Custom mode: You can edit this value
              </small>
            )}
          </SettingGroup>
        </ModalBody>

        <ModalFooter>
          <CancelButton onClick={onClose} disabled={loading}>
            Cancel
          </CancelButton>
          <SaveButton onClick={handleSave} disabled={loading}>
            {loading ? 'Saving...' : 'Save Settings'}
          </SaveButton>
        </ModalFooter>
      </ModalContent>
    </ModalOverlay>
  );
};

export default SettingsModal;
