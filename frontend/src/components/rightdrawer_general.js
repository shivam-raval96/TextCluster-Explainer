import React, { useContext, useState, useEffect } from 'react';
import { styled, useTheme } from "@mui/material/styles";
import Box from "@mui/material/Box";
import Drawer from "@mui/material/Drawer";
import CssBaseline from "@mui/material/CssBaseline";
import Divider from "@mui/material/Divider";
import IconButton from "@mui/material/IconButton";
import MenuIcon from "@mui/icons-material/Menu";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import Explain from './explanation'
import Explore from './explore';
import { AppContext } from '../AppContext';

const drawerWidth = 300;



const DrawerHeader = styled("div")(({ theme }) => ({
  display: "flex",
  alignItems: "center",
  padding: theme.spacing(0, 1),
  // necessary for content to be below app bar
  ...theme.mixins.toolbar,
  justifyContent: "flex-start"
}));





export default function RightDrawer(props) {

  const appcontext = useContext(AppContext);
  const theme = useTheme();

  const handleDrawerOpen = () => {
    props.setOpen(props.open === true ? false : true)
    props.setOtherOpen(false)
  };

  const handleDrawerClose = () => {
    props.setOpen(false);
  };

  return (
    <Box sx={{ display: "flex" }}>
      <CssBaseline />
      <IconButton
        style={{ display: "block" }}
        onClick={handleDrawerOpen}
        edge="end"
        sx={{ mt: 2, ml: 0, mr: 0, ...(props.open) }}
      >

        <p style={{ fontSize: 16 }}>{props.type}</p>
      </IconButton>
      <Drawer
        sx={{
          width: 5,
          flexShrink: 0,
          "& .MuiDrawer-paper": {
            width: drawerWidth
          }
        }}
        variant="persistent"
        anchor="right"
        open={props.open}
      >
        <DrawerHeader>
          <IconButton onClick={handleDrawerClose}>
            {theme.direction === "rtl" ? (
              <ChevronLeftIcon />
            ) : (
              <ChevronRightIcon />
            )}
            {props.type}
          </IconButton>
        </DrawerHeader>
        <Divider />
        {
          props.type === "Explanation" ?
            <Explain data={appcontext.isinsidelasso} getexplain={appcontext.getexplain} selected={appcontext.lassoed} />
            :
            <Explore />
        }


        <Divider />

      </Drawer>
    </Box>
  );
}