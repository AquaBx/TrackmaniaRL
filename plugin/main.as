class SpeedSocket {
    Net::Socket@ socket;
    Net::Socket@ client;

    SpeedSocket() {
        @socket = Net::Socket();
        if (!socket.Listen('localhost', 12345)) {
            print("Failed to listen on port 12345");
        } else {
            print("Listening on port 12345");
        }
    }

    void Accept() {
        auto tclient = socket.Accept();

        if (tclient !is null) {
            @client = tclient;
            print("Client connected");
        }
    }

    void Update(CSceneVehicleVisState@ vis) {
        if (client !is null && client.IsReady() && client.Available() > 0) {

            uint8 code = client.ReadUint8();

            mat4 proj = Camera::GetProjectionMatrix();
            CHmsCamera@ cam = Camera::GetCurrent();

            vec2 topLeft = 1 - (cam.DrawRectMax + 1) / 2;

            int w = Draw::GetWidth();
            int h = Draw::GetHeight();

            vec2 g_displayPos = topLeft * vec2(w,h);

            float posx = g_displayPos.x;
            float posy = g_displayPos.y;

            uint maxUint = 4294967295;

            uint8 flags = 0;

            if (maxUint!=vis.RaceStartTime) {
                flags = flags + 1;
            }

            MemoryBuffer buffer();

            buffer.Write( w ); // 4
            buffer.Write( h ); // 4
            buffer.Write( int(posx) ); // 4
            buffer.Write( int(posy) ); // 4

            buffer.Write( vis.Position.x ); // 4
            buffer.Write( vis.Position.y ); // 4
            buffer.Write( vis.Position.z ); // 4
 
            buffer.Write( vis.Left.x ); // 4
            buffer.Write( vis.Left.y ); // 4
            buffer.Write( vis.Left.z ); // 4
 
            buffer.Write( vis.Up.x ); // 4
            buffer.Write( vis.Up.y ); // 4
            buffer.Write( vis.Up.z ); // 4
 
            buffer.Write( vis.Dir.x ); // 4
            buffer.Write( vis.Dir.y ); // 4
            buffer.Write( vis.Dir.z ); // 4

            buffer.Write( proj.xx ); // 4
            buffer.Write( proj.yx ); // 4
            buffer.Write( proj.zx ); // 4
            buffer.Write( proj.tx ); // 4
            buffer.Write( proj.xy ); // 4
            buffer.Write( proj.yy ); // 4
            buffer.Write( proj.zy ); // 4
            buffer.Write( proj.ty ); // 4
            buffer.Write( proj.xz ); // 4
            buffer.Write( proj.yz ); // 4
            buffer.Write( proj.zz ); // 4
            buffer.Write( proj.tz ); // 4
            buffer.Write( proj.xw ); // 4
            buffer.Write( proj.yw ); // 4
            buffer.Write( proj.zw ); // 4
            buffer.Write( proj.tw ); // 4

            buffer.Write(vis.FrontSpeed * 3.6f); // 4
            buffer.Write(flags); // 1
            
            buffer.Seek(0);
            if (!client.Write(buffer)) {
                client.Close();
                @client = null;
                throw("failed to send close frame");
            }
        }
    }
}

SpeedSocket@ speedSocket;

void Main(){
    @speedSocket = SpeedSocket();
}

void Render() {
    CSceneVehicleVisState@ visState = VehicleState::ViewingPlayerState();

    if (visState is null) {
        return;
    }
    
    speedSocket.Accept();
    speedSocket.Update(visState);
}

