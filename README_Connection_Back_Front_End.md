# README

## Overview

This document outlines the integration of a detection algorithm (backend) with a React-based frontend application. The system architecture is designed to enhance performance by combining the detection process with a Flask server, allowing the model to be loaded only once when the server starts. This approach significantly speeds up the detection process.

## Backend: Flask Server and Socket.IO Integration

### Key Components

- **Flask Server**: Manages incoming requests and serves as the primary interface for the frontend.
- **Socket.IO**: Used for real-time communication between the backend and frontend.
- **Detection Algorithm**: A separate module integrated into the Flask server that processes the detection tasks.

### Functionality

- **Running Detection**: 
  - Route: `/run-detection`
  - Method: GET
  - Function: `handle_run_detection`
  - Description: Starts a new thread for detection and sends the response back to the client.

- **Reset Map**: 
  - Route: `/reset-map`
  - Method: GET
  - Function: `reset_map`
  - Description: Emits a signal to reset the map view on the frontend.

- **Socket.IO Emissions**:
  - The server emits data to the frontend using Socket.IO. This includes detection results and other necessary information like `level`, `index`, and `other_street`.

### Detection Process

1. A new detection task is initiated via the `/run-detection` route.
2. The detection process runs in a separate thread to maintain server responsiveness.
3. Upon completion, the server emits the detection results using Socket.IO.
4. If an error occurs, the error message is captured and sent to the frontend.

## Frontend: React Application

### Key Components

- **React Context (`AiOutputContext`)**: Manages the global state of detection outputs.

React Hook (`useAiOutput`): Custom hook for accessing the detection output context.
- **Socket.IO Client**: Used to establish a connection with the backend server and receive real-time updates.

### Functionality

- **Context Provider (`AiOutputProvider`)**:
  - Encapsulates the state logic for the detection output.
  - Provides `aiOutput` (an array of detection results) and `setAiOutput` (a function to update `aiOutput`).

- **Handling Real-Time Data**:
  - The frontend listens for data emitted by the backend using Socket.IO.
  - Upon receiving data, it updates the `aiOutput` state with the new detection results.
  - Special handling is done for `map_reset` and `other_street` data to manipulate the output accordingly.

### Data Flow

1. The frontend establishes a connection with the Flask server using Socket.IO.
2. When detection data is emitted by the backend, the frontend's Socket.IO client captures this data.
3. The received data is processed and stored in the `aiOutput` context state.
4. Components consuming the `AiOutputContext` can access and display the latest detection data.
5. The frontend can request additional actions like map reset from the backend.

### Integration of Detection Algorithm and Frontend

The integration allows the frontend to display real-time detection results from the backend algorithm. The use of React Context and Hooks ensures a clean and efficient state management across the application. The Socket.IO integration facilitates seamless and immediate communication between the backend and frontend, enabling a dynamic and responsive user interface.

## Conclusion

This system architecture effectively combines a powerful detection algorithm with a modern React frontend, ensuring high performance and real-time data handling. The use of Flask and Socket.IO on the backend, along with React Context and Hooks on the frontend, creates a robust and scalable solution suitable for real-time detection applications.
